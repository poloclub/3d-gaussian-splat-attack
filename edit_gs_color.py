import torch
import numpy as np
import sys
import argparse
sys.path.append("submodules/gaussian-splatting")
from utils.sh_utils import RGB2SH
from scene.gaussian_model import GaussianModel


def update_splats_color(model, color_sh_coefficients):
    """
    Core function to update the Gaussian splat colors in the model.
    
    Parameters:
    - model: The GaussianModel object.
    - color_sh_coefficients: Tensor of SH coefficients for the colors. Shape [num_points, 1, 3].
    """
    # Get the number of points in the model
    num_points = model._xyz.shape[0]

    # Ensure the color SH coefficients are in the correct shape [num_points, 1, 3]
    assert color_sh_coefficients.shape == (num_points, 1, 3), \
        f"Shape mismatch: expected ({num_points}, 1, 3), got {color_sh_coefficients.shape}"

    # Optionally set all higher SH terms to zero (removing complex lighting effects)
    new_features_rest = torch.zeros_like(model._features_rest).cuda()

    # Update the model's SH coefficients
    with torch.no_grad():
        model._features_dc.copy_(color_sh_coefficients)
        model._features_rest.copy_(new_features_rest)

    print("Updated splat colors.")


def change_splats_color_to_grayscale(model):
    """
    Convert all Gaussian splat colors to grayscale by projecting existing colors to grayscale space.
    
    Parameters:
    - model: The GaussianModel object.
    """
    # Get the number of points in the model
    num_points = model._xyz.shape[0]

    # Get the current SH coefficients for the direct component (RGB channels)
    current_features_dc = model._features_dc.detach().cpu().numpy()  # Shape: [num_points, 1, 3]
    
    # Extract the RGB values from the SH direct component
    current_colors_rgb = current_features_dc[:, 0, :]  # Shape: [num_points, 3]

    # Convert RGB to Grayscale using the luminosity method
    grayscale_values = 0.2989 * current_colors_rgb[:, 0] + \
                       0.5870 * current_colors_rgb[:, 1] + \
                       0.1140 * current_colors_rgb[:, 2]  # Shape: [num_points]

    # Create grayscale RGB (R = G = B = grayscale_value)
    grayscale_rgb = np.stack([grayscale_values] * 3, axis=-1)  # Shape: [num_points, 3]

    # Convert grayscale RGB values back to SH coefficients
    grayscale_sh = RGB2SH(torch.tensor(grayscale_rgb).float().cuda())  # Shape: [num_points, 3]

    # Reshape the SH coefficients to match the required format [num_points, 1, 3]
    grayscale_features_dc = grayscale_sh.view(num_points, 1, 3)

    # Call the core function to update the splat colors to grayscale
    update_splats_color(model, grayscale_features_dc)

    print("Converted all splat colors to grayscale.")


def change_splats_color_to_sepia(model):
    """
    Convert all Gaussian splat colors to sepia tone by applying a sepia transformation to existing colors.
    
    Parameters:
    - model: The GaussianModel object.
    """
    num_points = model._xyz.shape[0]
    current_features_dc = model._features_dc.detach().cpu().numpy()  # Shape: [num_points, 1, 3]
    current_colors_rgb = current_features_dc[:, 0, :]  # Shape: [num_points, 3]

    # Define the sepia filter matrix
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    # Apply the sepia filter
    sepia_colors_rgb = np.dot(current_colors_rgb, sepia_matrix.T)
    # Clip values to the valid range [0, 1]
    sepia_colors_rgb = np.clip(sepia_colors_rgb, 0, 1)
    sepia_sh = RGB2SH(torch.tensor(sepia_colors_rgb).float().cuda())  # Shape: [num_points, 3]
    sepia_features_dc = sepia_sh.view(num_points, 1, 3)
    update_splats_color(model, sepia_features_dc)
    print("Converted all splat colors to sepia tone.")



def change_splats_color_to_single(model, new_color_rgb):
    """
    Change all Gaussian splat colors in the model to a single new color.
    
    Parameters:
    - model: The GaussianModel object.
    - new_color_rgb: The new target color as an RGB tuple (e.g., (1.0, 0.0, 0.0) for red).
    """
    # Convert the RGB color to SH coefficients using the existing utility function.
    new_color_sh = RGB2SH(torch.tensor(new_color_rgb).float().cuda())  # Shape [3]
    num_points = model._xyz.shape[0]
    new_features_dc = new_color_sh.view(1, 1, 3).repeat(num_points, 1, 1)
    update_splats_color(model, new_features_dc)


def change_splats_color_to_random(model):
    """
    Change all Gaussian splat colors to random colors.
    
    Parameters:
    - model: The GaussianModel object.
    """
    # Get the number of points in the model
    num_points = model._xyz.shape[0]

    # Generate random RGB colors for each splat (shape: [num_points, 3])
    random_colors_rgb = np.random.rand(num_points, 3)
    random_colors_sh = RGB2SH(torch.tensor(random_colors_rgb).float().cuda())  # Shape [num_points, 3]
    random_features_dc = random_colors_sh.view(num_points, 1, 3)
    update_splats_color(model, random_features_dc)


def main():
    parser = argparse.ArgumentParser(description="Edit Gaussian splatting colors in a point cloud.")
    parser.add_argument("input_file", type=str, help="Path to the input PLY file.")
    parser.add_argument("output_file", type=str, help="Path to the output PLY file.")
    parser.add_argument("color_choice", type=str, choices=["single", "random", "grayscale", "sepia"], help="Color choice for the splats.")
    parser.add_argument("--rgb", type=float, nargs=3, metavar=("R", "G", "B"), help="RGB values for single color choice.")

    args = parser.parse_args()

    gm = GaussianModel(sh_degree=3)
    gm.load_ply(args.input_file)

    if args.color_choice == "single":
        if args.rgb:
            change_splats_color_to_single(gm, tuple(args.rgb))
        else:
            print("Error: RGB values must be provided for single color choice.")
            return
    elif args.color_choice == "random":
        change_splats_color_to_random(gm)
    elif args.color_choice == "grayscale":
        change_splats_color_to_grayscale(gm)
    elif args.color_choice == "sepia":
        change_splats_color_to_sepia(gm)

    gm.save_ply(args.output_file)

if __name__ == "__main__":
    main()
