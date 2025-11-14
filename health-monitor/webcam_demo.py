"""
Real-time webcam demo for CLABSIGuard healthcare monitoring
Privacy-preserving: displays live feed but does not record
"""
import cv2
import numpy as np
import torch
import time
import argparse
from clabsi_guard import CLABSIGuard
from clabsi_guard_v2 import CLABSIGuardV2
from pretrained_heads import PretrainedDepthHead, PretrainedSegmentationHead, PretrainedKeypointsHead
from monitor import ComplianceMonitor, ViolationType


class WebcamDemo:
    """Real-time webcam demo with visualization"""

    def __init__(self, camera_id=0, input_size=(640, 480), target_fps=30, use_v2=True, fullscreen=True):
        self.camera_id = camera_id
        self.input_size = input_size  # (width, height)
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps  # Target time per frame
        self.model_version = "V2 (Pretrained)" if use_v2 else "V1 (ResNet50)"
        self.fullscreen = fullscreen
        self.screen_width = 2560  # 2K resolution width
        self.screen_height = 1440  # 2K resolution height

        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot access webcam")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])

        # Try to set webcam FPS
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        # Initialize model
        print(f"Loading CLABSIGuard {self.model_version}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.use_v2 = use_v2

        if use_v2:
            # Use actual pretrained models for immediate high-quality predictions
            print("  Loading pretrained heads...")
            self.depth_head = PretrainedDepthHead()
            self.seg_head = PretrainedSegmentationHead()
            self.kp_head = PretrainedKeypointsHead()
            self.model = None  # Not using backbone-based model for V2
        else:
            # Use V1 with ResNet50 backbone
            self.model = CLABSIGuard(pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Enable optimizations
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True

        # Initialize compliance monitor
        self.monitor = ComplianceMonitor(buffer_size=30, sanitizer_timeout=60.0)

        # FPS tracking
        self.fps_history = []
        self.frame_count = 0
        self.last_frame_time = time.time()

        print(f"Initialization complete! Target FPS: {target_fps}")

    def preprocess_frame(self, frame):
        """Convert OpenCV frame to model input tensor"""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if rgb.shape[:2] != (self.input_size[1], self.input_size[0]):
            rgb = cv2.resize(rgb, self.input_size)

        # Convert to tensor and normalize
        tensor = torch.from_numpy(rgb).float()
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        tensor = tensor / 255.0  # Normalize to [0, 1]

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def visualize_depth(self, depth_map):
        """Convert depth map to colorized visualization"""
        # depth_map: [1, 1, H, W] tensor
        depth = depth_map[0, 0].cpu().numpy()

        # Normalize to [0, 255]
        depth_vis = (depth * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        return depth_colored

    def visualize_segmentation(self, seg_map):
        """Convert segmentation to colored overlay"""
        # seg_map: [1, num_instances, H, W] tensor (instance masks from YOLO)
        seg = seg_map[0].cpu().numpy()

        h, w = seg.shape[1], seg.shape[2]
        seg_colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Colors for different instances
        colors = [
            [255, 0, 0],     # Instance 0 (red)
            [0, 255, 0],     # Instance 1 (green)
            [0, 0, 255],     # Instance 2 (blue)
            [255, 255, 0],   # Instance 3 (yellow)
            [255, 0, 255],   # Instance 4 (magenta)
            [0, 255, 255],   # Instance 5 (cyan)
            [255, 128, 0],   # Instance 6 (orange)
            [128, 0, 255]    # Instance 7 (purple)
        ]

        # Overlay each instance mask with its color
        for i in range(min(seg.shape[0], len(colors))):
            mask = seg[i] > 0.5  # Threshold at 0.5
            if mask.any():
                seg_colored[mask] = colors[i]

        return seg_colored

    def visualize_keypoints(self, keypoints_map):
        """Convert keypoint heatmaps to visualization"""
        # keypoints_map: [1, 21, H, W] tensor
        kp = keypoints_map[0].cpu().numpy()

        # Sum all heatmaps
        kp_sum = np.sum(kp, axis=0)

        # Normalize
        if kp_sum.max() > 0:
            kp_sum = kp_sum / kp_sum.max()

        # Convert to uint8
        kp_vis = (kp_sum * 255).astype(np.uint8)

        # Apply colormap
        kp_colored = cv2.applyColorMap(kp_vis, cv2.COLORMAP_HOT)

        return kp_colored

    def draw_status_panel(self, frame, fps, status, violations):
        """Draw status information panel on frame"""
        h, w = frame.shape[:2]

        # Scale font sizes for larger displays
        scale_factor = w / 1280  # Base scale for 1280px width
        font_scale_large = 0.8 * scale_factor
        font_scale_medium = 0.6 * scale_factor
        font_scale_small = 0.5 * scale_factor
        thickness_large = max(2, int(2 * scale_factor))
        thickness_medium = max(1, int(1.5 * scale_factor))

        # Status colors
        status_colors = {
            ViolationType.NONE: (0, 255, 0),      # Green
            ViolationType.CAUTION: (0, 255, 255), # Yellow
            ViolationType.WARNING: (0, 165, 255), # Orange
            ViolationType.CRITICAL: (0, 0, 255)   # Red
        }

        color = status_colors.get(status, (255, 255, 255))

        # FPS color: green if hitting target, yellow if close, red if far
        fps_diff = abs(fps - self.target_fps)
        if fps_diff <= 2:
            fps_color = (0, 255, 0)  # Green - hitting target
        elif fps_diff <= 5:
            fps_color = (0, 255, 255)  # Yellow - close
        else:
            fps_color = (0, 165, 255)  # Orange - off target

        # Draw semi-transparent panel
        panel_height = int(120 * scale_factor)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Positions scaled
        y_offset = int(30 * scale_factor)
        y_spacing = int(35 * scale_factor)

        # Draw FPS with target
        cv2.putText(frame, f"FPS: {fps:.1f} / {self.target_fps}", (int(10*scale_factor), y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, fps_color, thickness_large)

        # Draw status
        cv2.putText(frame, f"Status: {status.value}", (int(10*scale_factor), y_offset + y_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, color, thickness_large)

        # Draw device and model info
        device_text = "GPU" if self.device.type == 'cuda' else "CPU"
        cv2.putText(frame, f"Device: {device_text} | Model: {self.model_version}",
                   (int(10*scale_factor), y_offset + y_spacing*2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness_medium)

        # Draw recent violations
        if violations:
            y_offset = panel_height + int(30 * scale_factor)
            for v in violations[:3]:  # Show last 3
                cv2.putText(frame, f"{v.type.value}: {v.message[:40]}",
                           (int(10*scale_factor), y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, status_colors[v.type], thickness_medium)
                y_offset += int(25 * scale_factor)

        return frame

    def draw_keypoints_on_frame(self, frame, keypoints_map, threshold=0.3):
        """Draw keypoints directly on frame"""
        kp = keypoints_map[0].cpu().numpy()
        h, w = frame.shape[:2]

        # Find peaks in each keypoint heatmap
        for i in range(kp.shape[0]):
            heatmap = kp[i]
            max_val = heatmap.max()

            if max_val > threshold:
                max_idx = heatmap.argmax()
                y_norm, x_norm = np.unravel_index(max_idx, heatmap.shape)

                # Scale to frame coordinates
                x = int((x_norm / heatmap.shape[1]) * w)
                y = int((y_norm / heatmap.shape[0]) * h)

                # Draw keypoint
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Draw connections for hand skeleton (simplified)
                if i > 0 and i % 4 == 0:  # Connect some keypoints
                    prev_heatmap = kp[i-1]
                    prev_max = prev_heatmap.max()
                    if prev_max > threshold:
                        prev_idx = prev_heatmap.argmax()
                        prev_y_norm, prev_x_norm = np.unravel_index(prev_idx, prev_heatmap.shape)
                        prev_x = int((prev_x_norm / prev_heatmap.shape[1]) * w)
                        prev_y = int((prev_y_norm / prev_heatmap.shape[0]) * h)
                        cv2.line(frame, (prev_x, prev_y), (x, y), (0, 255, 255), 1)

        return frame

    def create_visualization_grid(self, frame, outputs):
        """Create 2x2 grid of visualizations"""
        h, w = frame.shape[:2]

        # Resize outputs to match frame size
        depth_vis = self.visualize_depth(outputs['depth'])
        depth_vis = cv2.resize(depth_vis, (w, h))

        seg_vis = self.visualize_segmentation(outputs['segmentation'])
        seg_vis = cv2.resize(seg_vis, (w, h))

        kp_vis = self.visualize_keypoints(outputs['keypoints'])
        kp_vis = cv2.resize(kp_vis, (w, h))

        # Create frame with keypoints drawn
        frame_with_kp = frame.copy()
        frame_with_kp = self.draw_keypoints_on_frame(frame_with_kp, outputs['keypoints'])

        # Create 2x2 grid
        half_w = w // 2
        half_h = h // 2

        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Top-left: Original frame with keypoints
        grid[0:h, 0:w] = frame_with_kp

        # Top-right: Depth
        grid[0:h, w:w*2] = depth_vis

        # Bottom-left: Segmentation
        grid[h:h*2, 0:w] = seg_vis

        # Bottom-right: Keypoints heatmap
        grid[h:h*2, w:w*2] = kp_vis

        # Add labels (scaled for visibility)
        label_scale = 0.7 * (w / 640)  # Scale based on quadrant width
        label_thickness = max(2, int(2 * (w / 640)))
        label_y = int(30 * (w / 640))

        cv2.putText(grid, "Input + Keypoints", (int(10*(w/640)), label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)
        cv2.putText(grid, "Depth", (int((w + 10)*(w/640)), label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)
        cv2.putText(grid, "Segmentation", (int(10*(w/640)), int((h + 30)*(w/640))),
                   cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)
        cv2.putText(grid, "Keypoint Heatmaps", (int((w + 10)*(w/640)), int((h + 30)*(w/640))),
                   cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)

        return grid

    def create_overlay_view(self, frame, outputs):
        """Create single view with all overlays"""
        vis_frame = frame.copy()

        # Draw keypoints
        vis_frame = self.draw_keypoints_on_frame(vis_frame, outputs['keypoints'])

        # Overlay segmentation (semi-transparent)
        seg_vis = self.visualize_segmentation(outputs['segmentation'])
        seg_vis = cv2.resize(seg_vis, (frame.shape[1], frame.shape[0]))
        vis_frame = cv2.addWeighted(vis_frame, 0.7, seg_vis, 0.3, 0)

        # Add depth as side bar
        depth_vis = self.visualize_depth(outputs['depth'])
        depth_vis = cv2.resize(depth_vis, (frame.shape[1]//4, frame.shape[0]))

        # Concatenate depth on right side
        vis_frame = np.hstack([vis_frame, depth_vis])

        return vis_frame

    def run(self):
        """Main loop for real-time demo"""
        print("\n" + "="*50)
        print("CLABSIGuard Webcam Demo")
        print("="*50)
        print(f"Target FPS: {self.target_fps}")
        print(f"Resolution: {self.screen_width}x{self.screen_height} (2K)")
        print(f"Fullscreen: {self.fullscreen}")
        print("Press 'q' to quit")
        print("Press 's' to toggle simple/grid view")
        print("Press 'o' to toggle overlay mode")
        print("Press 'r' to reset monitor")
        print("Press 'f' to toggle fullscreen")
        print("="*50 + "\n")

        # Create named window
        window_name = 'CLABSIGuard Monitor'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Set fullscreen if requested
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Default to grid view to show all outputs continuously
        show_grid = True
        show_overlay = False

        try:
            while True:
                frame_start = time.time()

                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Model inference
                if self.use_v2:
                    # V2: Use pretrained model heads directly on numpy image
                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (self.input_size[1], self.input_size[0]):
                        rgb = cv2.resize(rgb, self.input_size)

                    # Run pretrained models
                    depth_np = self.depth_head.predict(rgb)  # (H, W)
                    seg_np = self.seg_head.predict(rgb)      # (8, H, W)
                    kp_np = self.kp_head.predict(rgb)        # (21, H, W)

                    # Convert to tensors for visualization compatibility
                    outputs = {
                        'depth': torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                        'segmentation': torch.from_numpy(seg_np).unsqueeze(0),          # [1, 8, H, W]
                        'keypoints': torch.from_numpy(kp_np).unsqueeze(0),              # [1, 21, H, W]
                        'detection': torch.zeros(1, 27, 60, 80)  # Placeholder
                    }
                else:
                    # V1: Use model with tensor input
                    input_tensor = self.preprocess_frame(frame)
                    with torch.no_grad():
                        outputs = self.model(input_tensor)

                # Update monitor
                violations = self.monitor.update(outputs)
                status = self.monitor.get_current_status()
                recent_violations = self.monitor.get_recent_violations(window_seconds=10.0)

                # Calculate actual FPS (smoothed)
                current_time = time.time()
                actual_frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time

                instant_fps = 1.0 / actual_frame_time if actual_frame_time > 0 else 0
                self.fps_history.append(instant_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)

                # Display smoothed FPS
                avg_fps = np.mean(self.fps_history)

                # Create visualization
                if show_grid:
                    vis_frame = self.create_visualization_grid(frame, outputs)
                    vis_frame = self.draw_status_panel(vis_frame, avg_fps, status, recent_violations)
                elif show_overlay:
                    vis_frame = self.create_overlay_view(frame, outputs)
                    vis_frame = self.draw_status_panel(vis_frame, avg_fps, status, recent_violations)
                else:
                    vis_frame = frame.copy()
                    vis_frame = self.draw_status_panel(vis_frame, avg_fps, status, recent_violations)

                # Scale to screen resolution for fullscreen
                if self.fullscreen:
                    vis_frame = cv2.resize(vis_frame, (self.screen_width, self.screen_height))

                # Display
                cv2.imshow(window_name, vis_frame)

                # Handle keyboard input (1ms wait for responsiveness)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    show_grid = not show_grid
                    if show_grid:
                        show_overlay = False
                    print(f"View mode: {'Grid (all outputs)' if show_grid else 'Simple'}")
                elif key == ord('o'):
                    show_overlay = not show_overlay
                    if show_overlay:
                        show_grid = False
                    print(f"Overlay mode: {'ON' if show_overlay else 'OFF'}")
                elif key == ord('r'):
                    self.monitor.reset()
                    print("Monitor reset")
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print(f"Fullscreen: {'ON' if self.fullscreen else 'OFF'}")

                self.frame_count += 1

                # FPS limiting: sleep to maintain target FPS
                processing_time = time.time() - frame_start
                sleep_time = self.frame_time - processing_time

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Show warning if we can't maintain target FPS
                if processing_time > self.frame_time and self.frame_count % 100 == 0:
                    print(f"Warning: Processing time ({processing_time*1000:.1f}ms) exceeds target ({self.frame_time*1000:.1f}ms)")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # Cleanup
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        print("\n" + "="*50)
        print("Session Statistics")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        if self.fps_history:
            print(f"Average FPS: {np.mean(self.fps_history):.1f}")
            print(f"Min FPS: {np.min(self.fps_history):.1f}")
            print(f"Max FPS: {np.max(self.fps_history):.1f}")

        summary = self.monitor.get_violation_summary()
        print("\nViolation Summary:")
        for v_type, count in summary.items():
            print(f"  {v_type.value}: {count}")
        print("="*50)


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description='CLABSIGuard Webcam Demo')
    parser.add_argument('--model', choices=['v1', 'v2'], default='v2',
                      help='Model version: v1 (ResNet50) or v2 (YOLO pretrained, default)')
    parser.add_argument('--fps', type=int, default=30,
                      help='Target FPS (default: 30)')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera ID (default: 0)')
    parser.add_argument('--fullscreen', action='store_true', default=True,
                      help='Start in fullscreen mode (default: True)')
    parser.add_argument('--no-fullscreen', action='store_false', dest='fullscreen',
                      help='Start in windowed mode')
    parser.add_argument('--width', type=int, default=2560,
                      help='Screen width for fullscreen (default: 2560 for 2K)')
    parser.add_argument('--height', type=int, default=1440,
                      help='Screen height for fullscreen (default: 1440 for 2K)')

    args = parser.parse_args()

    try:
        # Initialize with specified parameters
        use_v2 = (args.model == 'v2')
        demo = WebcamDemo(
            camera_id=args.camera,
            input_size=(640, 480),
            target_fps=args.fps,
            use_v2=use_v2,
            fullscreen=args.fullscreen
        )
        demo.screen_width = args.width
        demo.screen_height = args.height
        demo.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
