# How this project works, in plain English

## 1. What it is (say this first)

The Stretch 3 is a home robot from Hello Robot: a wheeled base, a tall mast with a lift that
moves up and down, an arm that telescopes out sideways, and a small wrist with a gripper. This
project computes how to move **all of these together** so the gripper reaches a point, and puts
a safety layer in between that never lets a command break the robot's limits.

**Honest status:** everything runs in simulation. Nothing has run on a real Stretch.

## 2. The five ideas

1. **Forward kinematics:** given every joint's position, where is the gripper? It is a chain of
   transforms: base, then mast, then lift height, then arm extension, then the three wrist angles.
   Each step is a 4x4 matrix (a rotation plus a shift), and multiplying them gives the gripper's
   position.
2. **Inverse kinematics:** the reverse. "The gripper should be *here*; which joint positions get
   it there?" There is no formula, so it is solved step by step: look at how each joint moves the
   gripper (the *Jacobian*), step in the direction that shrinks the error, repeat.
3. **Whole-body:** solve for base and arm together instead of "drive first, then reach", which
   often parks the base slightly wrong.
4. **Safety filter:** whatever a controller asks for, this layer finds the closest command that
   keeps every joint inside its limits, respects motor speeds, and keeps the robot from tipping
   (an arm far out at full height). It is a small optimisation solved every control cycle.
5. **Behaviour cloning:** a neural network trained to copy the IK controller. It learns the rough
   motion but not the last few centimetres, and it breaks joint limits, which is exactly why the
   safety filter must stay in the loop.

## 3. What was added before the interview, and why it matters

- **The real geometry.** The original model was a simplified stand-in. A second model now uses
  the real Stretch 3 dimensions from MuJoCo's official model library and agrees with MuJoCo to
  about a hundred-billionth of a metre. It showed that the original model had the arm on the
  wrong side.
- **Confidence intervals.** "993 of 1,000 targets solved" now comes with an exact 95 % interval
  (98.6–99.7 %), so a difference between solvers can be told apart from luck.
- **A ROS 2 node in C++.** It reads a target position through tf2 (ROS's system for keeping track
  of coordinate frames), computes the next step and filters it, 50 times a second. Its tests check
  that the C++ gives the same numbers as the Python. It is built and tested automatically on
  GitHub; it has not run on a robot.

## 4. The weak spot, said honestly

The model lets the base slide sideways. A real Stretch has two driven wheels: it can drive forward
and backward and turn, but not slide. So on the real robot, "slide left" becomes "turn, drive,
turn back". The repository now has the piece that removes sideways motion
(`differential_drive_map`), but the solvers do not use it yet. If asked, say exactly that.

## 5. Likely questions, honest answers

1. **"What is inverse kinematics and why is it hard here?"** Finding joint positions for a gripper
   position. Here the base and the arm can produce the same motion, so there are infinitely many
   answers and the maths has to cope with that.
2. **"Why did the simplest solver win?"** The damped methods give up accuracy to stay stable near
   these redundant poses; Jacobian transpose just steps downhill and loses nothing, it only needs
   more steps.
3. **"What does the safety filter guarantee?"** Joint limits and motor speeds, exactly, at every
   step. The tipping constraint can be relaxed when the robot already starts unsafe, and it is a
   simplified stand-in, not a measured stability model.
4. **"How did you check the kinematics?"** Against MuJoCo's official Stretch 3 model, on 1,000
   random poses, and the Jacobian against finite differences.
5. **"Has it run on the real robot?"** No. Simulation only; the next steps would be the
   differential-drive constraint and Hello Robot's ROS 2 interfaces.
6. **"Did you write this yourself?"** "With a lot of help from AI coding tools; I went through how
   each part works."
