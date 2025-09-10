from albumentations import DualTransform, ImageOnlyTransform
import numpy as np
import random
import cv2
from utils.util_fun import grouped_powerset

class FrequencyPatterns(DualTransform):
    def __init__(self, p=1.0, max_multi=3, weight_range=None):
        super(FrequencyPatterns, self).__init__(p=p)
        self.weight_range = weight_range
        self.max_multi = max_multi
        self.patterns = [
            self.pattern_grid,
            self.pattern_symmetric_grid,
            self.pattern_checkerdboard,
            self.pattern_circular_checkerboard,
            self.pattern_squares,
            self.pattern_random_lines,
            self.pattern_stripes,
        ]

    def apply(self, img, return_pattern=False, **kwargs):
        self.weight = np.random.uniform(self.weight_range[0], self.weight_range[1])
        pattern_functions = np.random.choice(self.patterns, size=np.random.randint(1, self.max_multi + 1), replace=True)
        pattern = None
        for pattern_function in pattern_functions:
            res = pattern_function(cols=img.shape[1], rows=img.shape[0])
            pattern = res if pattern is None else (pattern + res)
        f_pattern = np.fft.fft2(pattern, s=(img.shape[0], img.shape[1]))

        img_fft = np.fft.fft2(img, axes=(0, 1))
        magnitude_original = np.abs(img_fft)
        phase_original = np.angle(img_fft)
        magnitude_pattern = np.abs(f_pattern)[..., None]
        magnitude_result = (1 - self.weight) * magnitude_original + self.weight * magnitude_pattern
        f_result = magnitude_result * np.exp(1j * phase_original)
        result = np.fft.ifft2(f_result, axes=(0, 1)).real

        if np.max(result) > 255:
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        if return_pattern:
            return result, pattern
        return result

    def pattern_circular_checkerboard(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_sectors = random.randint(4, 16)
        center_x, center_y = cols // 2, rows // 2
        max_radius = min(center_x, center_y)
        if max_radius == 0:
            max_radius = 1
        sector_angle = 2 * np.pi / num_sectors
        for y in range(rows):
            for x in range(cols):
                dx = x - center_x
                dy = y - center_y
                angle = np.arctan2(dy, dx)
                if angle < 0:
                    angle += 2 * np.pi
                sector_index = int(angle / sector_angle)
                if sector_index % 2 == 0:
                    pattern[y, x] = 255
        return pattern

    def pattern_grid(self, cols, rows):
        pattern = np.zeros((rows, cols))
        cell_size = random.randint(10, 30)
        intensity_range = (50, 200)
        for y in range(0, rows, cell_size):
            for x in range(0, cols, cell_size):
                cell_intensity = random.randint(*intensity_range)
                pattern[y:y + cell_size, x:x + cell_size] = cell_intensity
        return pattern

    def pattern_checkerdboard(self, cols, rows):
        pattern = np.zeros((rows, cols))
        cell_size = random.randint(10, 30)
        intensity = random.randint(150, 255)
        center_x, center_y = cols // 2, rows // 2
        max_offset = min(center_x, center_y) - cell_size
        for y in range(center_y - max_offset, center_y + max_offset, cell_size * 2):
            for x in range(center_x - max_offset, center_x + max_offset, cell_size * 2):
                pattern[y:y + cell_size, x:x + cell_size] = intensity
        return pattern

    def pattern_squares(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_cells = random.randint(5, 20)
        cell_size = random.randint(10, 30)
        intensity = random.randint(150, 255)
        center_or_edge = random.choice(['center', 'edge'])
        for _ in range(num_cells):
            if center_or_edge == 'center':
                cell_x, cell_y = self.random_position_in_center(cols, rows, cell_size)
            else:
                cell_x, cell_y = self.random_position_in_edge(cols, rows, cell_size)
            pattern[cell_y:cell_y + cell_size, cell_x:cell_x + cell_size] = intensity
        return pattern

    def random_position_in_center(self, cols, rows, cell_size):
        center_x, center_y = cols // 2, rows // 2
        max_offset = min(center_x, center_y) - cell_size
        cell_x = random.randint(center_x - max_offset, center_x + max_offset - cell_size)
        cell_y = random.randint(center_y - max_offset, center_y + max_offset - cell_size)
        return cell_x, cell_y

    def random_position_in_edge(self, cols, rows, cell_size):
        edge_x = random.choice([0, cols - 1])
        edge_y = random.choice([0, rows - 1])
        cell_x = random.randint(edge_x, cols - cell_size) if edge_x == 0 else random.randint(0, cols - cell_size)
        cell_y = random.randint(edge_y, rows - cell_size) if edge_y == 0 else random.randint(0, rows - cell_size)
        return cell_x, cell_y

    def pattern_random_lines(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_lines = random.randint(5, 20)
        for _ in range(num_lines):
            line_intensity = random.randint(50, 200)
            line_length = random.randint(10, min(cols, rows) // 2)
            line_angle = random.uniform(0, 2 * np.pi)
            start_x = cols // 2
            start_y = rows // 2
            for i in range(line_length):
                x = int(start_x + i * np.cos(line_angle))
                y = int(start_y + i * np.sin(line_angle))
                if 0 <= x < cols and 0 <= y < rows:
                    pattern[y, x] = line_intensity
        return pattern

    def pattern_symmetric_grid(self, cols, rows):
        pattern = np.zeros((rows, cols))
        value_x = np.random.uniform(1, 50)
        value_y = value_x
        for y in range(rows):
            for x in range(cols):
                value = 128 + 127 * np.sin(2 * np.pi * value_x * x / cols) * np.sin(2 * np.pi * value_y * y / rows)
                pattern[y, x] = value
        return pattern

    def pattern_stripes(self, cols, rows):
        pattern = np.zeros((rows, cols))
        stripe_direction = random.choice(['horizontal', 'vertical', 'both'])
        if stripe_direction == 'both':
            vertical_stripes = self.pattern_stripes_single_direction(cols, rows, 'vertical')
            horizontal_stripes = self.pattern_stripes_single_direction(cols, rows, 'horizontal')
            pattern = np.maximum(vertical_stripes, horizontal_stripes)
        else:
            pattern = self.pattern_stripes_single_direction(cols, rows, stripe_direction)
        return pattern

    def pattern_stripes_single_direction(self, cols, rows, direction):
        pattern = np.zeros((rows, cols))
        num_stripes = random.randint(5, 20)
        stripe_width = random.randint(10, 30)
        stripe_distance = random.randint(30, 60)
        intensity = random.randint(100, 255)
        if direction == 'horizontal':
            for _ in range(num_stripes):
                start_y = random.randint(0, rows - 1)
                pattern[start_y:start_y + stripe_width, :] = intensity
                start_y += stripe_distance
        elif direction == 'vertical':
            for _ in range(num_stripes):
                start_x = random.randint(0, cols - 1)
                pattern[:, start_x:start_x + stripe_width] = intensity
                start_x += stripe_distance
        return pattern


class FrequencyArtifactGenerator(ImageOnlyTransform):
    def __init__(self, weight_range=(0.025, 0.1), max_multi=3, p=1.0):
        super(FrequencyArtifactGenerator, self).__init__(p)
        self.fp = FrequencyPatterns(p=p, weight_range=weight_range, max_multi=max_multi)
        self.mask_types = grouped_powerset([0, [8, 9], 13, 17, 18])
        
    @property
    def targets_as_params(self):
        return ["image", "label"]

    def get_params_dependent_on_targets(self, params):
        return {"label": params.get("label", None)}

    def apply(self, img, **params):
        label = params.get("label", None)
        mask = np.ones(label.shape, dtype=np.uint8)
        for masked_area in random.choice(self.mask_types):
            mask = mask & (label != masked_area)
        mask = mask[..., None]
        return self.fp.apply(img) * mask + img * (1.0 - mask)