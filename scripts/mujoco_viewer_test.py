import mujoco
import numpy as np
import imageio.v2 as imageio
import os

# Fix WSL window positioning issues
os.environ['SDL_VIDEO_WINDOW_POS'] = '200,200'  # Force window position
os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center windows

xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 0.1">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Create a renderer with offscreen buffer
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
pixels = renderer.render()

# Save the frame to PNG
imageio.imwrite("frame.png", pixels)
print("Saved frame.png")

# Optionally: open it with Windows image viewer (if WSLg works)
import subprocess
subprocess.run(["xdg-open", "frame.png"])
