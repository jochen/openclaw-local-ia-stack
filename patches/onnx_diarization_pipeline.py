# pyright: reportUnreachable=false
import logging
import time

from einops import rearrange
import numpy as np
import numpy.typing as npt
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from onnx_diarization.clustering import VBxClustering
from onnx_diarization.clustering.plda import PLDATransform
from onnx_diarization.embedding import WeSpeakerEmbeddingModel
from onnx_diarization.segmentation import PyannnoteSegmentation
from onnx_diarization.utils import aggregate, trim

logger = logging.getLogger(__name__)


class SpeakerDiarizationPipeline:
    def __init__(
        self,
        segmentation: PyannnoteSegmentation,
        embedding: WeSpeakerEmbeddingModel,
        plda: PLDATransform,
        clustering_threshold: float = 0.6,
        embedding_batch_size: int = 128,
        embedding_exclude_overlap: bool = False,
    ) -> None:
        self.segmentation = segmentation

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.plda = plda

        metric = self.embedding.metric

        self.clustering = VBxClustering(self.plda, metric=metric, threshold=clustering_threshold)

    def __call__(
        self,
        waveform: npt.NDArray[np.float32],
        file_id: str | None = None,
        known_speakers: dict[str, npt.NDArray[np.float32]] | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Annotation:
        logger.debug(f"Processing audio with {len(waveform)} samples")
        t_total_start = time.perf_counter()

        if file_id is None:
            file_id = "audio"

        min_speakers, max_speakers = validate_speakers_args(
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_known_speakers=len(known_speakers) if known_speakers else None,
        )

        t_seg_start = time.perf_counter()
        segmentations = self.segmentation(waveform)
        t_seg = time.perf_counter() - t_seg_start

        logger.info(
            f"Segmentation (multilabel): shape={segmentations.data.shape}, min={np.min(segmentations.data):.4f}, max={np.max(segmentations.data):.4f}"
        )
        logger.info(f"Active speakers per chunk: {np.sum(segmentations.data > 0, axis=(1, 2))}")

        count = self._speaker_count(segmentations, self.segmentation.frames, warm_up=(0.0, 0.0))

        if np.nanmax(count.data) == 0.0:
            logger.warning("No speakers detected in audio")
            return Annotation(uri=file_id)

        t_embed_start = time.perf_counter()
        embeddings = self._get_embeddings(waveform, segmentations)
        t_embed = time.perf_counter() - t_embed_start

        t_cluster_start = time.perf_counter()
        hard_clusters, _soft_clusters, centroids = self.clustering(
            embeddings=embeddings,
            segmentations=segmentations,
            num_clusters=None,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
        )
        t_cluster = time.perf_counter() - t_cluster_start

        num_different_speakers = np.max(hard_clusters) + 1
        logger.info(f"Detected {num_different_speakers} speakers")

        cluster_labels = None
        if known_speakers is not None:
            known_embeddings = self._extract_known_speaker_embeddings(known_speakers)
            cluster_labels = self._match_clusters_to_known_speakers(centroids, known_embeddings)

        if max_speakers is not None:
            count.data = np.minimum(count.data, max_speakers).astype(np.int8)
        else:
            count.data = count.data.astype(np.int8)

        inactive_speakers = np.sum(segmentations.data, axis=1) == 0
        hard_clusters[inactive_speakers] = -2

        t_reconstruct_start = time.perf_counter()
        discrete_diarization = self._reconstruct(segmentations, hard_clusters, count)
        annotation = self._to_annotation(
            discrete_diarization, min_duration_off=0.0, threshold=0.1, cluster_labels=cluster_labels
        )
        t_reconstruct = time.perf_counter() - t_reconstruct_start
        logger.info(f"Reconstruction took {t_reconstruct:.3f}s")

        annotation.uri = file_id

        t_total = time.perf_counter() - t_total_start
        logger.info(
            f"Total processing time: {t_total:.3f}s - "
            f"segmentation: {t_seg:.3f}s ({t_seg / t_total * 100:.1f}%), "
            f"embedding: {t_embed:.3f}s ({t_embed / t_total * 100:.1f}%), "
            f"clustering: {t_cluster:.3f}s ({t_cluster / t_total * 100:.1f}%), "
            f"reconstruction: {t_reconstruct:.3f}s ({t_reconstruct / t_total * 100:.1f}%)"
        )
        logger.info(f"Diarization complete: {len(annotation.labels())} speakers")
        return annotation

    def _speaker_count(
        self,
        binarized_segmentations: SlidingWindowFeature,
        frames: SlidingWindow,
        warm_up: tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        trimmed = trim(binarized_segmentations, warm_up=warm_up)
        count = aggregate(
            np.sum(trimmed, axis=-1, keepdims=True),
            frames,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)

        return count

    def _get_embeddings(
        self, audio_data: npt.NDArray[np.float32], segmentations: SlidingWindowFeature
    ) -> npt.NDArray[np.float32]:
        from onnx_diarization.embedding import EmbeddingSegment

        num_chunks = segmentations.data.shape[0]

        fbank_data = self.embedding.preprocess(audio_data)

        segments: list[EmbeddingSegment] = []
        for segment, mask_data in segmentations:
            mask_data_f32 = mask_data.astype(np.float32)
            for speaker_mask in mask_data_f32.T:
                mask = speaker_mask if np.sum(speaker_mask) > 0 else np.ones_like(speaker_mask)
                segments.append(EmbeddingSegment(start=segment.start, end=segment.end, mask=mask))

        all_embeddings = self.embedding.extract(fbank_data, segments, batch_size=self.embedding_batch_size)
        embeddings = rearrange(all_embeddings, "(c s) d -> c s d", c=num_chunks)

        return embeddings

    def _extract_known_speaker_embeddings(
        self, known_speakers: dict[str, npt.NDArray[np.float32]]
    ) -> dict[str, npt.NDArray[np.float32]]:
        from onnx_diarization.embedding import EmbeddingSegment

        logger.info(f"Extracting embeddings for {len(known_speakers)} known speakers")
        known_embeddings = {}

        for speaker_label, waveform in known_speakers.items():
            fbank_data = self.embedding.preprocess(waveform)

            audio_duration = len(waveform) / fbank_data["sample_rate"]
            mask = np.ones(int(audio_duration / fbank_data["frame_step_seconds"]), dtype=np.float32)

            segment = EmbeddingSegment(start=0.0, end=audio_duration, mask=mask)
            embeddings = self.embedding.extract(fbank_data, [segment], batch_size=1)

            known_embeddings[speaker_label] = embeddings[0]
            logger.debug(f"Extracted embedding for {speaker_label}: shape={embeddings[0].shape}")

        return known_embeddings

    def _match_clusters_to_known_speakers(
        self,
        centroids: npt.NDArray[np.float32],
        known_embeddings: dict[str, npt.NDArray[np.float32]],
        threshold: float = 0.5,
    ) -> dict[int, str]:
        logger.info(f"Matching {len(centroids)} clusters to {len(known_embeddings)} known speakers")

        known_labels = list(known_embeddings.keys())
        known_emb_array = np.array([known_embeddings[label] for label in known_labels])

        distances = cdist(centroids, known_emb_array, metric=self.embedding.metric)

        num_clusters = len(centroids)
        num_known = len(known_labels)

        if num_clusters == 0 or num_known == 0:
            return {}

        cost_matrix = distances[:num_clusters, :num_known]

        # PATCH (ai-stack, 2026-07-06): a cluster with zero valid training
        # embeddings (e.g. very short/degenerate segment) produces a NaN
        # centroid upstream, which turns into NaN/inf entries here and makes
        # linear_sum_assignment raise "matrix contains invalid numeric
        # entries" -> 500 on /v1/audio/diarization. Same guard already used
        # in clustering/vbx.py::constrained_argmax for the same reason.
        cost_matrix = np.nan_to_num(cost_matrix, nan=1e10, posinf=1e10, neginf=-1e10)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        cluster_labels = {}
        for cluster_idx, known_idx in zip(row_ind, col_ind, strict=False):
            distance = cost_matrix[cluster_idx, known_idx]
            similarity = 1 - distance if self.embedding.metric == "cosine" else -distance

            if similarity >= threshold:
                cluster_labels[cluster_idx] = known_labels[known_idx]
                logger.info(
                    f"Matched cluster {cluster_idx} to known speaker '{known_labels[known_idx]}' "
                    f"(similarity={similarity:.3f}, distance={distance:.3f})"
                )
            else:
                logger.debug(
                    f"Cluster {cluster_idx} did not match known speaker '{known_labels[known_idx]}' "
                    f"(similarity={similarity:.3f} < threshold={threshold})"
                )

        logger.info(f"Successfully matched {len(cluster_labels)} out of {num_clusters} clusters to known speakers")
        return cluster_labels

    def _reconstruct(
        self, segmentations: SlidingWindowFeature, hard_clusters: npt.NDArray[np.int32], count: SlidingWindowFeature
    ) -> SlidingWindowFeature:
        num_chunks, num_frames, _local_num_speakers = segmentations.data.shape

        valid_clusters = hard_clusters[hard_clusters >= 0]
        if len(valid_clusters) == 0:
            logger.warning("No valid clusters found, returning empty annotation")
            num_clusters = 0
        else:
            num_clusters = np.max(valid_clusters) + 1

        clustered_segmentations = np.zeros((num_chunks, num_frames, num_clusters), dtype=np.float32)

        for c, (cluster, (_chunk, segmentation)) in enumerate(zip(hard_clusters, segmentations, strict=False)):
            for k in np.unique(cluster):
                if k < 0:
                    continue

                mask = cluster == k
                if np.any(mask):
                    clustered_segmentations[c, :, k] = np.max(segmentation[:, mask], axis=1)

        clustered_segmentations = SlidingWindowFeature(clustered_segmentations, segmentations.sliding_window)

        return self._to_diarization(clustered_segmentations, count)

    def _to_diarization(self, segmentations: SlidingWindowFeature, count: SlidingWindowFeature) -> SlidingWindowFeature:
        activations = aggregate(
            segmentations,
            count.sliding_window,
            hamming=False,
            missing=0.0,
            skip_average=True,
        )

        _, num_speakers = activations.data.shape
        max_speakers_per_frame = np.max(count.data)
        if num_speakers < max_speakers_per_frame:
            activations.data = np.pad(activations.data, ((0, 0), (0, max_speakers_per_frame - num_speakers)))

        extent = activations.extent & count.extent
        activations = activations.crop(extent, return_data=False)
        count = count.crop(extent, return_data=False)

        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations.data)

        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers, strict=False)):
            for i in range(c.item()):
                binary[t, speakers[i]] = 1.0

        return SlidingWindowFeature(binary, activations.sliding_window)

    def _to_annotation(
        self,
        discrete_diarization: SlidingWindowFeature,
        min_duration_off: float = 0.0,  # noqa: ARG002
        threshold: float = 0.5,
        cluster_labels: dict[int, str] | None = None,
    ) -> Annotation:
        annotation = Annotation(uri="audio")

        for k, k_th_speaker in enumerate(discrete_diarization.data.T):
            label = cluster_labels.get(k, f"SPEAKER_{k:02d}") if cluster_labels else f"SPEAKER_{k:02d}"

            active = k_th_speaker > threshold
            if not np.any(active):
                continue

            active_indices = np.where(active)[0]
            starts = active_indices[np.r_[True, np.diff(active_indices) > 1]]
            ends = active_indices[np.r_[np.diff(active_indices) > 1, True]]

            for start_idx, end_idx in zip(starts, ends, strict=True):
                start_time = discrete_diarization.sliding_window[start_idx].start
                end_time = discrete_diarization.sliding_window[end_idx].end
                annotation[Segment(start_time, end_time)] = label

        return annotation


def validate_speakers_args(
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    num_known_speakers: int | None = None,
) -> tuple[int | None, int | None]:
    if min_speakers is not None and max_speakers is not None:
        if min_speakers > max_speakers:
            raise ValueError("min_speakers must be less than or equal to max_speakers")

    if num_known_speakers is not None:
        if num_known_speakers < 1:
            raise ValueError("num_known_speakers must be at least 1")

        if min_speakers is not None and min_speakers < num_known_speakers:
            raise ValueError(f"min_speakers ({min_speakers}) must be >= num_known_speakers ({num_known_speakers})")

        if max_speakers is not None and max_speakers < num_known_speakers:
            raise ValueError(f"max_speakers ({max_speakers}) must be >= num_known_speakers ({num_known_speakers})")

        if min_speakers is None:
            min_speakers = num_known_speakers

    return min_speakers, max_speakers
