# Audit, 7 Oct 2026

What was checked before extending the repository, and what changed.

## Reproduced

- All 52 existing tests pass on Windows 10, Python 3.13.
- `examples/01_solver_benchmark.py` and `examples/02_safety_benchmark.py` print exactly the
  tables quoted in the README.

## Found

1. **The geometry was a stand-in.** `StretchModel` uses plausible dimensions, not the robot's.
   Compared with MuJoCo Menagerie's Stretch 3, the arm extends to the wrong side (+y instead of
   -y), and the wrist geometry differs (the real gripper hangs below the wrist and points along
   the arm at zero yaw). No parameter change can fix that; the chain itself differs.
2. **The base is modelled as holonomic.** x, y and theta are independent joints, so the base can
   slide sideways. A real Stretch has a differential drive and cannot. The README's central
   observation, that base sliding and arm extension produce the same tool motion, is a property of
   this model, not of the robot. Neither the README nor the docstrings said so.
3. **No ROS 2 node, no MuJoCo comparison**, as the README itself stated.
4. **Wording that read as hardware experience.** The status line ("personal project, rebuilt and
   published") did not say that nothing has run on a robot.

## Changed

- `chain.py`: `Stretch3Chain`, the real geometry as 4x4 transforms from Menagerie, with the same
  interface as `StretchModel`, and `differential_drive_map` for the non-holonomic base.
- `tests/test_chain_mujoco.py`: forward kinematics against MuJoCo on 500 random configurations
  (agreement to about 4x10^-11 m), the geometric Jacobian against finite differences, the arm
  direction, the differential-drive map.
- `examples/05_exact_geometry.py`: the MuJoCo agreement and the IK benchmark on the real geometry
  with Clopper-Pearson intervals.
- `ros2/stretch_wbc_ros`: a C++ ROS 2 Humble node (tf2 target, Jacobian-transpose step, the
  existing C++ safety filter) with gtest against generated Python reference values; built and
  tested in CI only.
- README: honest status, the two models, the holonomic assumption, sections 6 and 7.
- `EXPLAIN.md`: plain-English explanation.

## Not done

- The solvers, filter and benchmarks of sections 1-4 still run on the simplified model.
- No custom ros2_control controller; the ROS 2 node simulates the robot itself.
- Nothing has run on a real Stretch or in RViz on the author's machine.
