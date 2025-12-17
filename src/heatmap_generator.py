"""
Heatmap Generator for Person Detection Data
Generates density heatmaps from detection data and overlays on floor plans.
"""

import cv2
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Colormaps available (OpenCV compatible)
COLORMAPS = {
    'jet': cv2.COLORMAP_JET,
    'hot': cv2.COLORMAP_HOT,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'plasma': cv2.COLORMAP_PLASMA,
    'inferno': cv2.COLORMAP_INFERNO,
    'magma': cv2.COLORMAP_MAGMA,
    'turbo': cv2.COLORMAP_TURBO,
    'rainbow': cv2.COLORMAP_RAINBOW,
    'ocean': cv2.COLORMAP_OCEAN,
    'cool': cv2.COLORMAP_COOL,
    'spring': cv2.COLORMAP_SPRING,
    'summer': cv2.COLORMAP_SUMMER,
    'autumn': cv2.COLORMAP_AUTUMN,
    'winter': cv2.COLORMAP_WINTER,
    'bone': cv2.COLORMAP_BONE,
}


class HeatmapGenerator:
    """
    Generate heatmaps from person detection data.
    Creates density visualizations overlaid on floor plans.
    """
    
    def __init__(
        self,
        plan_path: str,
        colormap: str = 'jet',
        alpha: float = 0.6,
        gaussian_sigma: float = 20.0,
        min_opacity: float = 0.0
    ):
        """
        Initialize the heatmap generator.
        
        Args:
            plan_path: Path to floor plan image
            colormap: Colormap name (jet, hot, viridis, etc.)
            alpha: Heatmap overlay transparency (0.0 - 1.0)
            gaussian_sigma: Sigma for Gaussian smoothing
            min_opacity: Minimum opacity for heatmap (areas with no data)
        """
        self.plan_path = Path(plan_path)
        self.colormap_name = colormap.lower()
        self.alpha = max(0.0, min(1.0, alpha))
        self.gaussian_sigma = gaussian_sigma
        self.min_opacity = min_opacity
        
        # Load floor plan
        self.plan_image = self._load_plan()
        self.height, self.width = self.plan_image.shape[:2]
        
        # Get colormap
        if self.colormap_name not in COLORMAPS:
            print(f"[WARN] Unknown colormap '{colormap}', using 'jet'")
            self.colormap_name = 'jet'
        self.colormap = COLORMAPS[self.colormap_name]
        
        print(f"[INFO] Heatmap generator initialized:")
        print(f"   - Plan size: {self.width}x{self.height}")
        print(f"   - Colormap: {self.colormap_name}")
        print(f"   - Alpha: {self.alpha}")
        print(f"   - Gaussian sigma: {self.gaussian_sigma}")
    
    def _load_plan(self) -> np.ndarray:
        """Load floor plan image"""
        if not self.plan_path.exists():
            raise FileNotFoundError(f"Floor plan not found: {self.plan_path}")
        
        image = cv2.imread(str(self.plan_path))
        if image is None:
            raise ValueError(f"Failed to load floor plan: {self.plan_path}")
        
        print(f"[OK] Loaded floor plan: {self.plan_path}")
        return image
    
    def load_detections(self, detection_path: str) -> List[Tuple[float, float]]:
        """
        Load detection data from JSON file.
        
        Args:
            detection_path: Path to detection JSON file
            
        Returns:
            List of (x, y) plan coordinates
        """
        detection_path = Path(detection_path)
        
        if not detection_path.exists():
            raise FileNotFoundError(f"Detection file not found: {detection_path}")
        
        with open(detection_path, 'r') as f:
            data = json.load(f)
        
        detections = data.get('detections', [])
        points = []
        
        for det in detections:
            plan_point = det.get('plan_point')
            if plan_point and len(plan_point) == 2:
                x, y = plan_point
                # Filter out points outside the plan bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    points.append((x, y))
                else:
                    # Points outside bounds - still include but will be clipped
                    points.append((
                        max(0, min(x, self.width - 1)),
                        max(0, min(y, self.height - 1))
                    ))
        
        total = len(data.get('detections', []))
        print(f"[OK] Loaded {len(points)}/{total} valid detection points")
        
        return points
    
    def create_density_map(
        self,
        points: List[Tuple[float, float]],
        bin_size: int = 1
    ) -> np.ndarray:
        """
        Create a 2D density map from points.
        
        Args:
            points: List of (x, y) coordinates
            bin_size: Size of histogram bins (1 = pixel-level)
            
        Returns:
            2D density array (float)
        """
        if not points:
            print("[WARN] No points provided for density map")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        # Calculate histogram bins
        x_bins = self.width // bin_size
        y_bins = self.height // bin_size
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Create 2D histogram
        density, x_edges, y_edges = np.histogram2d(
            y_coords, x_coords,  # Note: y first for proper array indexing
            bins=[y_bins, x_bins],
            range=[[0, self.height], [0, self.width]]
        )
        
        # Resize to full resolution if binned
        if bin_size > 1:
            density = cv2.resize(
                density.astype(np.float32),
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR
            )
        
        print(f"[INFO] Density map created: max={density.max():.0f}, total={density.sum():.0f}")
        
        return density.astype(np.float32)
    
    def smooth_density(self, density: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to density map using OpenCV.
        
        Args:
            density: 2D density array
            
        Returns:
            Smoothed density array
        """
        if self.gaussian_sigma > 0:
            # Convert sigma to kernel size (must be odd)
            ksize = int(self.gaussian_sigma * 6) | 1  # Ensure odd
            ksize = max(3, ksize)  # Minimum kernel size
            smoothed = cv2.GaussianBlur(density, (ksize, ksize), self.gaussian_sigma)
            print(f"[INFO] Applied Gaussian smoothing (sigma={self.gaussian_sigma}, ksize={ksize})")
            return smoothed
        return density
    
    def normalize_density(self, density: np.ndarray) -> np.ndarray:
        """
        Normalize density to 0-255 range.
        
        Args:
            density: 2D density array
            
        Returns:
            Normalized uint8 array
        """
        if density.max() > 0:
            normalized = (density / density.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(density, dtype=np.uint8)
        
        return normalized
    
    def apply_colormap(self, density: np.ndarray) -> np.ndarray:
        """
        Apply colormap to normalized density.
        
        Args:
            density: Normalized uint8 density array
            
        Returns:
            BGR colored heatmap
        """
        colored = cv2.applyColorMap(density, self.colormap)
        return colored
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        density: np.ndarray = None
    ) -> np.ndarray:
        """
        Overlay heatmap on floor plan.
        
        Args:
            heatmap: Colored heatmap (BGR)
            density: Original density for alpha masking (optional)
            
        Returns:
            Composited image
        """
        # Ensure same size
        if heatmap.shape[:2] != self.plan_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (self.width, self.height))
        
        # Create alpha mask based on density
        if density is not None:
            # Normalize density for alpha mask
            if density.max() > 0:
                alpha_mask = density / density.max()
            else:
                alpha_mask = np.zeros_like(density)
            
            # Apply minimum opacity threshold
            alpha_mask = np.clip(alpha_mask, self.min_opacity, 1.0)
            
            # Scale by overall alpha
            alpha_mask = alpha_mask * self.alpha
            
            # Expand to 3 channels
            alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
            
            # Blend
            result = (
                self.plan_image.astype(np.float32) * (1 - alpha_mask) +
                heatmap.astype(np.float32) * alpha_mask
            ).astype(np.uint8)
        else:
            # Simple alpha blend
            result = cv2.addWeighted(
                self.plan_image, 1 - self.alpha,
                heatmap, self.alpha,
                0
            )
        
        return result
    
    def generate(
        self,
        points: List[Tuple[float, float]],
        show_points: bool = False,
        point_radius: int = 3
    ) -> np.ndarray:
        """
        Generate complete heatmap visualization.
        
        Args:
            points: List of (x, y) coordinates
            show_points: Whether to draw individual points
            point_radius: Radius for point markers
            
        Returns:
            Final composited image
        """
        # Create density map
        density = self.create_density_map(points)
        
        # Smooth
        smoothed = self.smooth_density(density)
        
        # Normalize
        normalized = self.normalize_density(smoothed)
        
        # Apply colormap
        heatmap = self.apply_colormap(normalized)
        
        # Overlay on plan
        result = self.overlay_heatmap(heatmap, smoothed)
        
        # Optionally draw individual points
        if show_points:
            for x, y in points:
                cv2.circle(result, (int(x), int(y)), point_radius, (255, 255, 255), -1)
        
        print("[OK] Heatmap generated successfully")
        return result
    
    def save(self, image: np.ndarray, output_path: str) -> str:
        """
        Save heatmap image.
        
        Args:
            image: Image to save
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        print(f"[OK] Saved heatmap to: {output_path}")
        
        return str(output_path)
    
    def generate_from_file(
        self,
        detection_path: str,
        output_path: str = None,
        show_points: bool = False
    ) -> str:
        """
        Generate heatmap from detection file.
        
        Args:
            detection_path: Path to detection JSON file
            output_path: Output file path (auto-generated if None)
            show_points: Whether to draw individual points
            
        Returns:
            Path to saved file
        """
        # Load detections
        points = self.load_detections(detection_path)
        
        if not points:
            print("[WARN] No valid points found in detection file")
        
        # Generate heatmap
        result = self.generate(points, show_points=show_points)
        
        # Auto-generate output path if needed
        if output_path is None:
            detection_name = Path(detection_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"heatmap_{detection_name}_{timestamp}.png"
        
        # Save
        return self.save(result, output_path)


def find_plan_path(detection_path: str) -> Optional[str]:
    """
    Try to find floor plan path from detection data.
    
    Args:
        detection_path: Path to detection JSON file
        
    Returns:
        Path to floor plan or None
    """
    try:
        with open(detection_path, 'r') as f:
            data = json.load(f)
        
        plan_path = data.get('plan_path')
        if plan_path and Path(plan_path).exists():
            return plan_path
        
        # Try to find from store name
        store_name = data.get('metadata', {}).get('store_name')
        if store_name:
            calibration_dir = Path("calibration_data")
            for f in calibration_dir.glob(f"calibration_{store_name}*.json"):
                with open(f, 'r') as file:
                    cal_data = json.load(file)
                    plan_path = cal_data.get('store', {}).get('plan_path')
                    if plan_path and Path(plan_path).exists():
                        return plan_path
        
    except Exception as e:
        print(f"[WARN] Error finding plan path: {e}")
    
    return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate heatmap from person detection data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to detection JSON file"
    )
    parser.add_argument(
        "--plan", "-p",
        default=None,
        help="Path to floor plan image (auto-detected if not provided)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (auto-generated if not provided)"
    )
    parser.add_argument(
        "--colormap", "-c",
        default="jet",
        choices=list(COLORMAPS.keys()),
        help="Colormap for heatmap (default: jet)"
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.6,
        help="Heatmap transparency 0.0-1.0 (default: 0.6)"
    )
    parser.add_argument(
        "--sigma", "-s",
        type=float,
        default=20.0,
        help="Gaussian smoothing sigma (default: 20.0)"
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Draw individual detection points"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window before saving"
    )
    parser.add_argument(
        "--list-colormaps",
        action="store_true",
        help="List available colormaps and exit"
    )
    
    args = parser.parse_args()
    
    # List colormaps if requested
    if args.list_colormaps:
        print("Available colormaps:")
        for name in sorted(COLORMAPS.keys()):
            print(f"  - {name}")
        return 0
    
    try:
        # Find plan path if not provided
        plan_path = args.plan
        if plan_path is None:
            plan_path = find_plan_path(args.input)
            if plan_path is None:
                print("[ERROR] Could not find floor plan. Please provide --plan argument.")
                return 1
            print(f"[INFO] Auto-detected floor plan: {plan_path}")
        
        # Initialize generator
        generator = HeatmapGenerator(
            plan_path=plan_path,
            colormap=args.colormap,
            alpha=args.alpha,
            gaussian_sigma=args.sigma
        )
        
        # Load detections
        points = generator.load_detections(args.input)
        
        if not points:
            print("[ERROR] No valid detection points found")
            return 1
        
        # Generate heatmap
        result = generator.generate(points, show_points=args.show_points)
        
        # Show preview if requested
        if args.preview:
            print("[INFO] Showing preview (press any key to continue)")
            cv2.imshow("Heatmap Preview", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Determine output path
        output_path = args.output
        if output_path is None:
            detection_name = Path(args.input).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"heatmap_{timestamp}.png"
        
        # Save
        generator.save(result, output_path)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
