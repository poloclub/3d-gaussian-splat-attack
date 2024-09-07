# 3D-Gaussian-Splat-Attack


### Edit Gaussian Splat Colors
In `edit_gs_color.py`, you can edit the colors of the Gaussian splats in the point cloud. The script can be used to change the colors to a single color, grayscale, sepia, or random colors. The script can be used as follows:

```bash
python edit_gs_color.py input.ply output.ply single --rgb 0.0 0.0 1.0

python edit_gs_color.py input.ply output.ply grayscale

python edit_gs_color.py input.ply output.ply sepia

python edit_gs_color.py input.ply output.ply random
```