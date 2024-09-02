from screeninfo import get_monitors


class Config:
    FPS = 60  # Frames per Second.
    #Screen information
    # Get screen dimensions
    monitors = get_monitors()
    for monitor in monitors:
        WIDTH= monitor.width
        HEIGHT= monitor.height

    # Movement
    ACC = 1.2  # Acceleration- rate of change of velocity
    FRIC = -0.10  # Friction to reduce the velocity of an object.