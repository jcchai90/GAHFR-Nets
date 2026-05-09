from utils.visualization_utils import plot_cam, concat_images
import sys
import os


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def cam_dataset_model(sel_dataset, sel_model, n_classes, size=None):
    project_root = get_project_root()
    
    sys.argv.append('--common.config-file')
    sys.argv.append(os.path.join(project_root, 'config', 'classification', 'food_image', 'ehfr_net_food101.yaml'))

    sys.argv.append('--model.classification.pretrained')
    sys.argv.append(os.path.join(project_root, 'cam_relative_file', sel_dataset, sel_model, 'checkpoint_ema_best.pt'))

    sys.argv.append('--common.override-kwargs')
    sys.argv.append('model.classification.n_classes={}'.format(n_classes))

    plot_cam(image_path=os.path.join(project_root, 'cam_relative_file', sel_dataset, 'origin', '*.jpg'),
             cam_path=os.path.join(project_root, 'cam_relative_file', sel_dataset, sel_model, 'cam_results'), size=size)


def concatenate_images(
        sel_dataset,
        image_path=None,
        size=None,
        padding=None,
        num_column=4,
):
    project_root = get_project_root()
    if image_path is None:
        image_path = os.path.join(project_root, 'cam_relative_file', sel_dataset, 'total', '*.jpg')
    concat_images(image_path=image_path, num_column=num_column, size=size, padding=padding)


'''
    Before running the code, it is necessary to follow the CAM_ Relative_ Create a folder in the form of a file and 
    place it in the trained model file.
'''

# generative heat map
# sel_dataset: The name of the folder where the dataset is stored.
# sel_model: Model name.
# n_classes: Number of categories.
# size: Enter image size.
cam_dataset_model(sel_dataset="food101", sel_model="baseline", n_classes="101", size=(256, 256))

# concatenate images
# sel_dataset: The name of the folder where the dataset is stored.
# image_path: The position of the images that need to be spliced.
# concatenate_images(sel_dataset="food172", image_path=None, size=None,
#                    padding=10, num_column=2)
