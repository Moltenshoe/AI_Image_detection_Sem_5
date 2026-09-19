from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

# --------------------------------------------------
# HEIC SUPPORT
# --------------------------------------------------

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif is not installed.")
    print("HEIC images will not work.")
    print("Install it with:")
    print("pip install pillow-heif")


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FOLDER = BASE_DIR / "original_photos"
OUTPUT_FOLDER = BASE_DIR / "real"

OUTPUT_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# IMAGE SETTINGS
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif"
}

OUTPUT_SIZE = 512

MAX_DISPLAY_SIZE = 800

# Size of crop box relative to displayed image.
# It is FIXED and cannot be resized by the user.
CROP_RATIO = 0.75


# --------------------------------------------------
# FIND IMAGES
# --------------------------------------------------

if INPUT_FOLDER.exists():

    images = sorted(
        [
            path
            for path in INPUT_FOLDER.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower()
    )

else:
    images = []


# --------------------------------------------------
# VARIABLES
# --------------------------------------------------

current_index = 0

original_image = None
rotated_image = None

display_image = None
photo = None

rotation_angle = 0

crop_size = 0

crop_x = 0
crop_y = 0

drag_start_x = 0
drag_start_y = 0

dragging = False


# --------------------------------------------------
# CREATE WINDOW
# --------------------------------------------------

root = tk.Tk()

root.title("512 × 512 Dataset Cropper")


# --------------------------------------------------
# UI
# --------------------------------------------------

filename_label = tk.Label(
    root,
    text="",
    font=("Arial", 14)
)

filename_label.pack(pady=10)


canvas = tk.Canvas(
    root,
    highlightthickness=0
)

canvas.pack()


button_frame = tk.Frame(root)
button_frame.pack(pady=10)


rotate_left_button = tk.Button(
    button_frame,
    text="⟲ Rotate Left",
    command=lambda: rotate_image(-90),
    width=15
)

rotate_left_button.grid(
    row=0,
    column=0,
    padx=5
)


rotate_right_button = tk.Button(
    button_frame,
    text="⟳ Rotate Right",
    command=lambda: rotate_image(90),
    width=15
)

rotate_right_button.grid(
    row=0,
    column=1,
    padx=5
)


reset_button = tk.Button(
    button_frame,
    text="Reset Rotation",
    command=lambda: rotate_image(0),
    width=15
)

reset_button.grid(
    row=0,
    column=2,
    padx=5
)


save_button = tk.Button(
    button_frame,
    text="SAVE  (S)",
    command=lambda: save_crop(),
    width=15
)

save_button.grid(
    row=0,
    column=3,
    padx=5
)


skip_button = tk.Button(
    button_frame,
    text="SKIP  (X)",
    command=lambda: skip_image(),
    width=15
)

skip_button.grid(
    row=0,
    column=4,
    padx=5
)


instructions = tk.Label(
    root,
    text=(
        "Drag the FIXED crop box to position it\n"
        "S = Save    X = Skip    R = Reset Rotation    Q = Quit"
    ),
    font=("Arial", 12)
)

instructions.pack(pady=5)


# --------------------------------------------------
# UPDATE CROP BOX
# --------------------------------------------------

def draw_crop_box():

    canvas.delete("crop_box")

    canvas.create_rectangle(
        crop_x,
        crop_y,
        crop_x + crop_size,
        crop_y + crop_size,
        outline="red",
        width=4,
        tags="crop_box"
    )


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

def load_image():

    global original_image
    global rotated_image
    global display_image
    global photo

    global rotation_angle
    global crop_size
    global crop_x
    global crop_y

    if current_index >= len(images):

        canvas.delete("all")

        filename_label.config(
            text="Finished!"
        )

        instructions.config(
            text="All images have been processed."
        )

        return

    image_path = images[current_index]

    try:

        original_image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:

        print(
            f"Could not open {image_path.name}: {error}"
        )

        next_image()

        return

    rotation_angle = 0

    rotated_image = original_image.copy()

    # Create display copy
    display_image = rotated_image.copy()

    display_image.thumbnail(
        (MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(
        display_image
    )

    canvas.delete("all")

    canvas.config(
        width=display_image.width,
        height=display_image.height
    )

    canvas.create_image(
        0,
        0,
        anchor="nw",
        image=photo
    )

    # Fixed crop size
    crop_size = int(
        min(
            display_image.width,
            display_image.height
        ) * CROP_RATIO
    )

    # Center crop box
    crop_x = (
        display_image.width - crop_size
    ) // 2

    crop_y = (
        display_image.height - crop_size
    ) // 2

    draw_crop_box()

    filename_label.config(
        text=(
            f"{current_index + 1}/{len(images)}   "
            f"{image_path.name}   "
            f"{original_image.width} × "
            f"{original_image.height}"
        )
    )


# --------------------------------------------------
# ROTATE IMAGE
# --------------------------------------------------

def rotate_image(angle):

    global rotated_image
    global display_image
    global photo
    global rotation_angle

    if original_image is None:
        return

    # Reset rotation
    if angle == 0:

        rotation_angle = 0

        rotated_image = original_image.copy()

    else:

        rotation_angle = (
            rotation_angle + angle
        ) % 360

        rotated_image = original_image.rotate(
            -rotation_angle,
            expand=True
        )

    # Create display version
    display_image = rotated_image.copy()

    display_image.thumbnail(
        (MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(
        display_image
    )

    canvas.delete("all")

    canvas.config(
        width=display_image.width,
        height=display_image.height
    )

    canvas.create_image(
        0,
        0,
        anchor="nw",
        image=photo
    )

    # Recalculate fixed crop box
    global crop_size
    global crop_x
    global crop_y

    crop_size = int(
        min(
            display_image.width,
            display_image.height
        ) * CROP_RATIO
    )

    crop_x = (
        display_image.width - crop_size
    ) // 2

    crop_y = (
        display_image.height - crop_size
    ) // 2

    draw_crop_box()


# --------------------------------------------------
# START MOVING CROP BOX
# --------------------------------------------------

def start_drag(event):

    global drag_start_x
    global drag_start_y
    global dragging

    # Check whether click is inside crop box
    if (
        crop_x <= event.x <= crop_x + crop_size
        and
        crop_y <= event.y <= crop_y + crop_size
    ):

        dragging = True

        drag_start_x = event.x
        drag_start_y = event.y


# --------------------------------------------------
# MOVE CROP BOX
# --------------------------------------------------

def move_crop(event):

    global crop_x
    global crop_y
    global drag_start_x
    global drag_start_y

    if not dragging:
        return

    dx = event.x - drag_start_x
    dy = event.y - drag_start_y

    new_x = crop_x + dx
    new_y = crop_y + dy

    # Keep crop box inside image
    new_x = max(
        0,
        min(
            new_x,
            display_image.width - crop_size
        )
    )

    new_y = max(
        0,
        min(
            new_y,
            display_image.height - crop_size
        )
    )

    crop_x = new_x
    crop_y = new_y

    drag_start_x = event.x
    drag_start_y = event.y

    draw_crop_box()


# --------------------------------------------------
# STOP MOVING
# --------------------------------------------------

def stop_drag(event):

    global dragging

    dragging = False


# --------------------------------------------------
# SAVE CROP
# --------------------------------------------------

def save_crop():

    if rotated_image is None:
        return

    image_path = images[current_index]

    # Convert display coordinates
    # back to rotated original image coordinates.

    scale_x = (
        rotated_image.width
        / display_image.width
    )

    scale_y = (
        rotated_image.height
        / display_image.height
    )

    left = int(crop_x * scale_x)
    top = int(crop_y * scale_y)

    right = int(
        (crop_x + crop_size)
        * scale_x
    )

    bottom = int(
        (crop_y + crop_size)
        * scale_y
    )

    # Crop from original-resolution rotated image
    cropped = rotated_image.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    # EXACTLY 512 × 512
    cropped = cropped.resize(
        (
            OUTPUT_SIZE,
            OUTPUT_SIZE
        ),
        Image.Resampling.LANCZOS
    )

    output_path = (
        OUTPUT_FOLDER
        / f"{image_path.stem}.png"
    )

    cropped.save(
        output_path,
        "PNG",
        quality=95
    )

    print(
        f"Saved: {output_path.name} "
        f"(512 × 512)"
    )

    next_image()


# --------------------------------------------------
# SKIP
# --------------------------------------------------

def skip_image():

    print(
        f"Skipped: {images[current_index].name}"
    )

    next_image()


# --------------------------------------------------
# NEXT IMAGE
# --------------------------------------------------

def next_image():

    global current_index

    current_index += 1

    load_image()


# --------------------------------------------------
# KEYBOARD SHORTCUTS
# --------------------------------------------------

root.bind(
    "<s>",
    lambda event: save_crop()
)

root.bind(
    "<S>",
    lambda event: save_crop()
)

root.bind(
    "<x>",
    lambda event: skip_image()
)

root.bind(
    "<X>",
    lambda event: skip_image()
)

root.bind(
    "<r>",
    lambda event: rotate_image(0)
)

root.bind(
    "<R>",
    lambda event: rotate_image(0)
)

root.bind(
    "<q>",
    lambda event: root.destroy()
)

root.bind(
    "<Q>",
    lambda event: root.destroy()
)


# --------------------------------------------------
# MOUSE CONTROLS
# --------------------------------------------------

canvas.bind(
    "<ButtonPress-1>",
    start_drag
)

canvas.bind(
    "<B1-Motion>",
    move_crop
)

canvas.bind(
    "<ButtonRelease-1>",
    stop_drag
)


# --------------------------------------------------
# START PROGRAM
# --------------------------------------------------

if not INPUT_FOLDER.exists():

    print("\nERROR:")
    print(
        f"Input folder not found:\n"
        f"{INPUT_FOLDER}"
    )

    print("\nExpected structure:")
    print("Minor/")
    print("├── original_photos/")
    print("├── real/")
    print("└── crop_dataset.py")

    root.destroy()

elif not images:

    print("\nERROR:")
    print(
        f"No images found in:\n"
        f"{INPUT_FOLDER}"
    )

    root.destroy()

else:

    print(
        f"Found {len(images)} images."
    )

    print(
        f"Input : {INPUT_FOLDER}"
    )

    print(
        f"Output: {OUTPUT_FOLDER}"
    )

    load_image()

    root.mainloop()