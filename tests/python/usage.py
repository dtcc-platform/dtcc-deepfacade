import time, datetime
from deepfacade.unified_window_detection import unified_window_detection_multiple

start = time.time()

## Variant 1 (default)
predictions = unified_window_detection_multiple(
    images_directory_path="/home/sidkas/workspace/Facade2WindowPose/data/examples",
    auto_slice=True,
    save_predictions=True,
    plot_predictions=True
)

"""
## Variant 2
predictions = unified_window_detection_multiple(
    images_directory_path="/home/sidkas/workspace/Facade2WindowPose/data/examples",
    auto_slice=False,
    slice_image=True,
    slice_h=256,
    slice_w=512,
    save_predictions=True,
    plot_predictions=True
)

## Variant 3
predictions = unified_window_detection_multiple(
    images_directory_path="/home/sidkas/workspace/Facade2WindowPose/data/examples",
    slice_image=False,
    save_predictions=True,
    plot_predictions=True
)
"""

end = time.time()

print ("time elapsed: ", int(end - start), "(s)")

print(len(predictions.items()))