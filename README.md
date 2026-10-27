# FLUID-Parabola-Stabilization

UI-driven standalone reproduction of the retained V06 flight-video preparation.
It contains only the stabilization/correction code, retained V06 annotations,
and the fixed WR000001 calibration arrays. It does not contain chromatic-activity
analysis, plots, paper figures, or result tables.

1. Put the camera-calibrated N21 video at `input/N21_TWEEN_cam_cal.avi`, or select
   it in the UI.
2. Install `requirements.txt`.
3. Run `python run_ui.py`.
4. Select the video and output folder, then press **Run V06 preparation**.

The supplied `config.yaml` contains the settings retained by the V06 run. The UI
uses the bundled 4-point, residual 4-point, and 2-point annotations, then renders
4+4+2 plus the fixed white-reference correction from the original input video.
The expected quantitative output is
`output/final_v06/tracked_full_frame_rectified.mp4`.

The bundled annotations are specific to the 8097-frame N21 camera-calibrated
record. The runner refuses a video with a different frame size/count/FPS instead
of silently applying the V06 geometry to another recording.

