import numpy as np
import torch
from PIL import Image, ImageDraw
import re

from scipy.interpolate import BSpline


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class EmbeddingsCurveEditor:


    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "text": ("STRING", {"multiline": True, "default": "(0,1.0),(0.25,1.0),(0.75,1.0),(1.0,1.0)"}),
                "embed": ("EMBEDS",),

            }}

    RETURN_TYPES = ("IMAGE", "EMBEDS",)
    RETURN_NAMES = ("image", "embed",)
    FUNCTION = "apply"
    OUTPUT_NODE = True
    CATEGORY = "Chris's nodes"

    def bspline(self, length: int, pts: list) -> list:
        """
        Returns a list(range(length)) of normalized values of B-spline from control points pts.

        :param length: The length of the output list.
        :param pts: A list of control points.
        :return: A list of normalized B-spline values.
        """
        # Ensure we have at least 4 control points for a cubic B-spline
        if len(pts) < 4:
            raise ValueError("At least 4 control points are required")

        # Create a uniform knot vector
        k = 3  # Cubic B-spline
        n = len(pts)
        t = np.linspace(0, 1, n - k + 1)
        t = np.pad(t, (k, k), 'constant', constant_values=(0, 1))

        # Convert control points to a numpy array
        pts = np.array(pts)

        # Create the B-spline representation
        spline = BSpline(t, pts, k)

        # Evaluate the B-spline at evenly spaced intervals
        x = np.linspace(0, 1, length)
        y = spline(x)

        # Normalize the output values to be in the range [0, 1]
        y_min = np.min(y)
        y_max = np.max(y)
        normalized_y = (y - y_min) / (y_max - y_min)

        return normalized_y.tolist()

    def mul(lst, value):
        return [element * value for element in lst]

    def getControlPoints(self, input_string):
        result = list()
        # Regex pattern to match tuples of floats
        pattern = r"\((\d*\.\d+|\d+),(\d*\.\d+|\d+)\)"

        # Find all matches in the input string
        matches = re.findall(pattern, input_string)

        # Convert matched string tuples to float tuples
        result = [(float(x), float(y)) for x, y in matches]
        return result

    def drawEmbeddings(self, pos_embed, img, draw, color):
        embed_length = len(pos_embed[0])

        embed_values = pos_embed[0]

        pmax = pos_embed[0].max()
        prev_p = (0, 0)
        allpts = []
        for pi, p in enumerate(embed_values[1:]):
            allpts.append(prev_p)
            pint = int(p / pmax * 255.0)
            curr_p = (pi, pint)
            draw.line((prev_p, curr_p), fill=color, width=0)
            prev_p = curr_p
        return img

    def draw_bspline(self, weights, img, draw):
        prev_p = (0, int(np.round(weights[1] * 255)))
        for pi, p in enumerate(weights[1:], start=1):

            p = (pi, int(np.round(p * 255)))

            # Draw a line from the previous point to the current point
            draw.line([prev_p, p], fill="white", width=1)

            # Update the previous point to the current point
            prev_p = p

        return img



    def weights_from_bspline(self, ctrlpts, length):
        bspl_pts = self.bspline(length, pts=ctrlpts)
        return np.array(bspl_pts)[:, 1].tolist()





    def apply(self, text, embed):
        embed_values = embed[0]
        embed_length = len(embed[0])
        img = Image.new("RGB", size=(embed_length, 256))
        draw = ImageDraw.Draw(img)

        img = self.drawEmbeddings(embed, img, draw, "red")
        weights = self.weights_from_bspline(ctrlpts=self.getControlPoints(text), length=img.size[0])
        img = self.draw_bspline(weights=weights, img=img, draw=draw)
        new_embed = np.array(embed)*np.array(weights)
        img = self.drawEmbeddings(new_embed, img, draw, "blue")



        img = img.transpose(Image.FLIP_TOP_BOTTOM)


        return pil2tensor(img), torch.from_numpy(new_embed)
