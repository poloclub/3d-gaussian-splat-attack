# 3D-Gaussian-Splat-Attack


### Edit Gaussian Splat Colors
In `edit_gs_color.py`, you can edit the colors of the Gaussian splats in the point cloud. The script can be used to change the colors to a single color, grayscale, sepia, or random colors. The script can be used as follows:

```bash
python edit_gs_color.py input.ply output.ply single --rgb 0.0 0.0 1.0

python edit_gs_color.py input.ply output.ply grayscale

python edit_gs_color.py input.ply output.ply sepia

python edit_gs_color.py input.ply output.ply random
```

### Creating an attackable example 
The `splats` folder contains pre-trained Gaussian Splat scenes.  Using the original project, you need to combine both the input data (before creating splats) and the output data (trained splat scene) into a single folder as follows:
```
<location>
|---distorted
|   |---sparse
|       |---0
|           |---cameras.bin
|           |---images.bin
|           |---points3D.bin
|           |---project.ini
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---point_cloud
|   |---iteration_7000
|   |   |---point_cloud.ply
|   |---iteration_30000
|       |---point_cloud.ply
|---sparse
|   |---0
|       |---cameras.bin
|       |---images.bin
|       |---points3D.bin
|---stereo
|   |---consistency_graphs
|   |---depth_maps
|   |---normal_maps
|   |---fusion.cfg
|   |---patch-match.cfg
|---cameras.json
|---cfg_args
|---input.ply
|---run-colmap-geometric.sh
|---run-colmap-photometric.sh

```