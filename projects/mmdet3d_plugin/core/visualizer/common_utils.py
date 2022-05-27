import random


def get_color(index):
    """Get RGB color tuple
    """
    color_list = (
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Yellow green
        (0, 0, 255),    # Blue
        (0, 0, 128),    # Deep blue 
        (0, 255, 255),  # Light blue
        (128, 128, 0),  # Ocehr
        (0, 128, 0),    # Green
        (0, 128, 128),  # Blue green
        (128, 0, 128),  # Purple
        (255, 0, 255),  # Magenta
    )
    if index < len(color_list):
        color = color_list[index]
    else:
        color = tuple([random.choice(range(0, 255, 32)) for i in range(3)])
    return color
