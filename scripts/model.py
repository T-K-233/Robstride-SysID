"""Inlined MJCF for the single-actuator system-identification model.

A single hinge with `armature`, `damping`, and `frictionloss` as the unknowns.
A `motor` actuator with `gear="1"` so `ctrl[0]` is feed-forward output torque
in N.m. Sensors expose joint position and velocity. Generic across actuator
families -- the same model is used for any single-joint output-shaft test
rig. `armature` lumps rotor inertia + reflected gearbox inertia at the output;
the body's geometric inertia is kept tiny and fixed so all rotational inertia
is absorbed by that one parameter.
"""

import mujoco


JOINT_NAME = "shaft"
ACTUATOR_NAME = "torque"


ACTUATOR_XML = """\
<mujoco model="actuator">
  <compiler angle="radian" autolimits="false"/>
  <option integrator="implicitfast" timestep="0.0025">
    <flag contact="disable" gravity="disable"/>
  </option>

  <default>
    <joint type="hinge" axis="0 0 1" limited="false"/>
    <motor ctrllimited="true" ctrlrange="-17 17"/>
  </default>

  <worldbody>
    <body name="output_shaft" pos="0 0 0">
      <inertial pos="0 0 0" mass="0.20" diaginertia="1e-7 1e-7 1e-7"/>
      <joint name="shaft" armature="0.001" damping="0.05" frictionloss="0.1"/>
      <geom type="cylinder" size="0.03 0.005" rgba="0.6 0.6 0.65 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="torque" joint="shaft" gear="1"/>
  </actuator>

  <sensor>
    <jointpos name="position" joint="shaft"/>
    <jointvel name="velocity" joint="shaft"/>
  </sensor>
</mujoco>
"""


def make_spec(timestep: float) -> mujoco.MjSpec:
    """Return a fresh `MjSpec` for the actuator model with the given timestep.

    Each call produces an independent spec, so callers can freely set
    joint armature/damping/frictionloss without affecting other consumers.
    """
    spec = mujoco.MjSpec.from_string(ACTUATOR_XML)
    spec.option.timestep = float(timestep)
    return spec
