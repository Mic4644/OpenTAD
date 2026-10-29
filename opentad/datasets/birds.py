"""Bird behavior datasets for the annotations written by prepare_dataset.py.
    COPIES the structure of thumos datsets but not the thumos specific ambigious class

"""

from copy import deepcopy

import numpy as np

from .base import PaddingDataset, SlidingWindowDataset, filter_same_annotation
from .builder import DATASETS


def _get_bird_annotations(video_info, class_map, filter_gt, thresh):
    """Convert behavior intervals from seconds to frame indices and labels to IDs."""
    segments = []
    labels = []
    for annotation in video_info["annotations"]:
        gt_start = int(annotation["segment"][0] / video_info["duration"] * video_info["frame"])
        gt_end = int(annotation["segment"][1] / video_info["duration"] * video_info["frame"])

        if not filter_gt or gt_end - gt_start > thresh:
            segments.append([gt_start, gt_end])
            labels.append(class_map.index(annotation["label"]))

    if len(segments) == 0:  # have no valid gt
                return None

    return filter_same_annotation(
        dict(
            gt_segments=np.array(segments, dtype=np.float32),
            gt_labels=np.array(labels, dtype=np.int32),
        )
    )


@DATASETS.register_module()
class BirdsPaddingDataset(PaddingDataset):
    """Read bird crops for training; the pipeline handles truncation and padding.

    Inherits annotation-file loading, split selection, and category-file loading
    from PaddingDataset. Implements the bird annotations and per-video inputs.
    """

    def get_gt(self, video_info, thresh=0.0):
        return _get_bird_annotations(video_info, self.class_map, self.filter_gt, thresh)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            
            video_anno = deepcopy(video_anno) # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # convert to snippet indices for the current window
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=video_info["frame"] / video_info["duration"],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class BirdsSlidingDataset(SlidingWindowDataset):
    """Read bird crops as overlapping windows for validation and inference."""

    def get_gt(self, video_info, thresh=0.0):
        return _get_bird_annotations(video_info, self.class_map, self.filter_gt, thresh)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno) # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        return self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                fps=video_info["frame"] / video_info["duration"],
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
