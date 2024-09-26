import click

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.gamepad_teleop import GamePad
from stretch_mujoco.utils import display_camera_feeds


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--robocasa-env", is_flag=True, help="Use robocasa environment")
@click.option("--headless", is_flag=True, help="Run in headless mode")
def main(scene_xml_path: str, robocasa_env: bool, headless: bool):
    if robocasa_env:
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard()
        # breakpoint()
        robot_sim = StretchMujocoSimulator(model=model)
    elif scene_xml_path:
        robot_sim = StretchMujocoSimulator(scene_xml_path=scene_xml_path)
    else:
        robot_sim = StretchMujocoSimulator()
    gamepad = GamePad()
    robot_sim.start(headless=headless)
    gamepad.run_aync(robot_sim)

    gamepad.activate()
    display_camera_feeds(robot_sim)


if __name__ == "__main__":
    main()
