import os
import xml.etree.ElementTree as ET
from typing import Tuple

import mujoco
import robocasa.models
from robocasa.models.objects.kitchen_object_utils import (
    sample_kitchen_object,
    sample_kitchen_object_helper,
)
from robocasa.models.objects.objects import MJCFObject
from scipy.spatial.transform import Rotation as R

import stretch_mujoco.utils as utils
from stretch_mujoco import StretchMujocoSimulator

"""
object_group = ["all","cup","waffles","fruit",...]
"""


def create_obj(cfg: dict, obj_num: int = 0) -> Tuple[MJCFObject, dict]:
    """
    Create an object from the given configuration dictionary.
    Args:
        cfg (dict): Configuration dictionary
        obj_num (int): Object number needs to be unique if multiple objects are created
    """
    if "mjcf_path" in cfg:
        mjcf_path = cfg["mjcf_path"]
        # replace with correct base path
        new_base_path = os.path.join(robocasa.models.assets_root, "objects")
        new_path = os.path.join(new_base_path, mjcf_path.split("/objects/")[-1])
        obj_groups = new_path
        exclude_obj_groups = None
    else:
        obj_groups = cfg.get("obj_groups", "all")
        exclude_obj_groups = cfg.get("exclude_obj_groups", None)
    object_kwargs, object_info = sample_kitchen_object(
        obj_groups,
        exclude_groups=exclude_obj_groups,
        graspable=cfg.get("graspable", None),
        washable=cfg.get("washable", None),
        microwavable=cfg.get("microwavable", None),
        cookable=cfg.get("cookable", None),
        freezable=cfg.get("freezable", None),
        max_size=cfg.get("max_size", (None, None, None)),
        object_scale=cfg.get("object_scale", None),
    )
    if "name" not in cfg:
        cfg["name"] = "obj_{}".format(obj_num + 1)
    info = object_info

    object = MJCFObject(name=cfg["name"], **object_kwargs)

    return object, info


def extract_obj_asset_xml(xml: str) -> str:
    """
    Get the assets xml from the given xml string
    """
    root = ET.fromstring(xml)
    assets = root.find("asset")
    assets_xml = ET.tostring(assets, encoding="unicode")
    assets_lines = assets_xml.splitlines()
    assets_lines = [
        line
        for line in assets_lines
        if ("mesh" in line) or ("texture" in line) or ("material" in line)
    ]
    return assets_lines


def extract_object_body_xml(xml: str) -> str:
    """
    Get the object body xml from the given xml string
    """
    root = ET.fromstring(xml)
    body = root.find("worldbody/body/body")
    body_xml = ET.tostring(body, encoding="unicode")
    return body_xml


def insert_tree_into_worldbody(xml: str, tree_xml: str) -> str:
    """
    Insert a tree XML into the worldbody tag of the given XML string.
    """
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    tree_element = ET.fromstring(tree_xml)
    worldbody.append(tree_element)
    return ET.tostring(root, encoding="unicode")


def insert_assets_into_xml(xml: str, assets_lines: list) -> str:
    """
    Insert asset lines into the asset tag of the given XML string.
    """
    root = ET.fromstring(xml)
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")

    for line in assets_lines:
        asset_element = ET.fromstring(line)
        assets.append(asset_element)

    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    object_group = ["mug", "water_bottle"]  # ["coffee_cup","cup", "mug", "water_bottle"]
    kargs, info = sample_kitchen_object_helper(groups=object_group, graspable=True)
    print(kargs)
    print(info)
    obj, info = create_obj(kargs)
    obj_xml = obj.get_xml()
    asset_xml = extract_obj_asset_xml(obj_xml)
    body_xml = extract_object_body_xml(obj_xml)

    def euler_to_quat(euler_angles):
        r = R.from_euler("xyz", euler_angles)
        quat = r.as_quat()
        return tuple(quat)

    # Example usage
    euler_angles = (3.14, 0, 0)
    quat = euler_to_quat(euler_angles)
    robot_pose_attrib = {"pos": "0 0 0", "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"}
    stretch_xml_path = utils.get_absolute_path_stretch_xml(robot_pose_attrib)
    with open(f"{utils.models_path}/grasp_scene.xml", "r") as file:
        scene_model_xml = file.read()
    scene_model_xml = utils.insert_line_after_mujoco_tag(
        scene_model_xml,
        f' <include file="{stretch_xml_path}"/>',
    )

    scene_model_xml = insert_tree_into_worldbody(scene_model_xml, body_xml)
    scene_model_xml = insert_assets_into_xml(scene_model_xml, asset_xml)
    scene_model_xml = utils.xml_modify_body_pos(
        scene_model_xml, "body", "obj_1_main", [0, -0.65, 0.6], [0, 0, 0, 1]
    )
    scene_model = mujoco.MjModel.from_xml_string(scene_model_xml)
    # breakpoint()

    robot = StretchMujocoSimulator(model=scene_model)
    robot.start()
