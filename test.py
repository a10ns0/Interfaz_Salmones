
import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()
# Get and
print("ZED SDK Version:", zed.get_sdk_version())