# Example 67: Planar FK Pullback

This isolated folder explains the geometry behind the C-space manifolds used in
Examples 65 and 66 with a lower-dimensional 2-link planar robot.

The robot configuration is:

```text
theta = [theta1, theta2]
```

Forward kinematics maps that configuration to the end-effector point:

```text
x(theta) = l1*cos(theta1) + l2*cos(theta1 + theta2)
y(theta) = l1*sin(theta1) + l2*sin(theta1 + theta2)
```

The single-constraint demo defines a task-space constraint, such as a circle or
a line:

```text
h(x,y) = 0
```

Then it pulls that constraint back through forward kinematics:

```text
F(theta) = h(FK(theta))
```

The C-space manifold is the zero set:

```text
M = { theta | F(theta) = 0 }
```

Because this demo has 2 configuration variables and 1 scalar equality
constraint, the result is a 1D curve in 2D C-space.

## Why The C-Space Curve Is Not Manually Drawn

The C-space curve is extracted from the actual residual field
`F(theta)=h(FK(theta))` using matplotlib contour extraction. The script samples
theta-space on a grid, evaluates the pulled-back residual at each point, and
draws the zero contour. No artificial C-space curve is inserted.

This mirrors the thesis-facing Examples 65 and 66:

- Example 67: 2D C-space plus 1 scalar constraint gives a 1D curve.
- Examples 65/66: 3D C-space plus 1 scalar constraint gives a 2D surface.

## Single-Constraint Demo

Script:

```text
examples/ex67_planar_pullback/example_67_planar_2d_cspace_pullback_demo.py
```

It shows one task-space circle or line and its pulled-back C-space curve.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B examples\ex67_planar_pullback\example_67_planar_2d_cspace_pullback_demo.py --constraint circle --grid-res 400 --save-figure outputs\example_67_planar_pullback.png
```

For a line constraint:

```powershell
.\.venv\Scripts\python.exe -B examples\ex67_planar_pullback\example_67_planar_2d_cspace_pullback_demo.py --constraint line --grid-res 400 --save-figure outputs\example_67_planar_pullback_line.png
```

## Multimodal Demo

Script:

```text
examples/ex67_planar_pullback/example_67b_planar_multimodal_pullback_demo.py
```

This is the small 2D analogue of Example 66. It uses three task-space
constraints:

- left circle
- transfer line
- right circle

Each one is pulled back through FK:

```text
F_left(theta)  = h_left(FK(theta))
F_line(theta)  = h_line(FK(theta))
F_right(theta) = h_right(FK(theta))
```

The plotted C-space manifolds are the zero contours:

```text
M_left  = { theta | F_left(theta)=0 }
M_line  = { theta | F_line(theta)=0 }
M_right = { theta | F_right(theta)=0 }
```

The transfer line in task space is not copied into C-space as a straight line.
It becomes whatever curve satisfies `h_line(FK(theta))=0`.

The analogy to Example 66 is:

```text
2D demo:
curve -> transition point -> curve -> transition point -> curve

3D real example:
surface -> transition curve -> surface -> transition curve -> surface
```

Run:

```powershell
.\.venv\Scripts\python.exe -B examples\ex67_planar_pullback\example_67b_planar_multimodal_pullback_demo.py --grid-res 420 --save-figure outputs\example_67b_planar_multimodal_pullback.png
```

Use `--no-show` with either demo to save the figure without opening a
matplotlib window.
