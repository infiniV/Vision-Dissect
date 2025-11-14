"""
ComplianceMonitor: Simple temporal monitoring with rule-based violation detection
No complex state machines - just frame-by-frame checks over temporal buffer
"""
from collections import deque
from enum import Enum
import time


class ViolationType(Enum):
    """Violation severity levels"""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    CAUTION = "CAUTION"
    NONE = "NONE"


class Violation:
    """Simple violation record"""
    def __init__(self, violation_type, message, timestamp):
        self.type = violation_type
        self.message = message
        self.timestamp = timestamp

    def __repr__(self):
        return f"[{self.type.value}] {self.message} (t={self.timestamp:.1f}s)"


class ComplianceMonitor:
    """
    Temporal monitoring with simple rule-based violation detection
    Maintains 30-frame buffer for context
    """
    def __init__(self, buffer_size=30, sanitizer_timeout=60.0):
        self.buffer_size = buffer_size
        self.sanitizer_timeout = sanitizer_timeout

        # Temporal buffer for recent frames
        self.frame_buffer = deque(maxlen=buffer_size)

        # Tracking state
        self.last_sanitizer_time = None
        self.violations = []
        self.start_time = time.time()

    def update(self, predictions):
        """
        Update monitor with new frame predictions
        Returns current violations detected

        predictions: dict with keys:
            - detection: [batch, anchors*(5+classes), h, w]
            - segmentation: [batch, num_classes, h, w]
            - depth: [batch, 1, h, w]
            - keypoints: [batch, num_keypoints, h, w]
        """
        current_time = time.time() - self.start_time

        # Add to temporal buffer
        frame_data = {
            'timestamp': current_time,
            'predictions': predictions
        }
        self.frame_buffer.append(frame_data)

        # Check violations
        violations = []

        # Rule 1: Bare hand in sterile zone (CRITICAL)
        bare_hand_violation = self._check_bare_hand_in_sterile_zone(predictions)
        if bare_hand_violation:
            violations.append(bare_hand_violation)

        # Rule 2: No sanitizer interaction in timeout period (WARNING)
        sanitizer_violation = self._check_sanitizer_timeout(predictions, current_time)
        if sanitizer_violation:
            violations.append(sanitizer_violation)

        # Rule 3: Person proximity without PPE (CAUTION)
        proximity_violation = self._check_person_proximity(predictions)
        if proximity_violation:
            violations.append(proximity_violation)

        # Store violations
        self.violations.extend(violations)

        return violations

    def _check_bare_hand_in_sterile_zone(self, predictions):
        """
        Rule: Bare hand detected in sterile zone area
        Returns Violation if detected, None otherwise
        """
        # Simplified logic: Check if bare_hand detected AND sterile_zone segmented
        # In real implementation, would check spatial overlap

        # Detection classes: [bare_hand=0, gloved_hand=1, person=2, sanitizer=3]
        # Segmentation classes: [person=0, sterile_zone=1, equipment=2, floor=3, ...]

        detection = predictions['detection']
        segmentation = predictions['segmentation']

        # Disabled for now - need proper detection parsing
        # Real version would parse detection output and check spatial overlap
        has_bare_hand = False  # Would parse detection output
        has_sterile_zone = False  # Would check segmentation mask

        if has_bare_hand and has_sterile_zone:
            return Violation(
                ViolationType.CRITICAL,
                "Bare hand detected in sterile zone",
                time.time() - self.start_time
            )

        return None

    def _check_sanitizer_timeout(self, predictions, current_time):
        """
        Rule: No sanitizer interaction detected within timeout period
        Returns Violation if timeout exceeded, None otherwise
        """
        # Detection classes: [bare_hand=0, gloved_hand=1, person=2, sanitizer=3]

        detection = predictions['detection']

        # Simplified: Check if sanitizer (class 3) is detected
        # Real version would parse detection output properly

        has_sanitizer_interaction = False  # Placeholder

        if has_sanitizer_interaction:
            self.last_sanitizer_time = current_time

        # Check timeout
        if self.last_sanitizer_time is None:
            time_since_sanitizer = current_time
        else:
            time_since_sanitizer = current_time - self.last_sanitizer_time

        if time_since_sanitizer > self.sanitizer_timeout:
            return Violation(
                ViolationType.WARNING,
                f"No hand sanitization in {time_since_sanitizer:.0f}s (limit: {self.sanitizer_timeout:.0f}s)",
                current_time
            )

        return None

    def _check_person_proximity(self, predictions):
        """
        Rule: Person detected too close without proper PPE
        Returns Violation if detected, None otherwise
        """
        # Detection classes: [bare_hand=0, gloved_hand=1, person=2, sanitizer=3]

        detection = predictions['detection']
        depth = predictions['depth']

        # Simplified: Check if person detected (class 2) with shallow depth
        # Real version would:
        # 1. Parse person bboxes from detection
        # 2. Check depth values in those regions
        # 3. Verify gloved hands present (PPE check)

        has_close_person = False  # Placeholder
        has_gloves = False  # Placeholder

        if has_close_person and not has_gloves:
            return Violation(
                ViolationType.CAUTION,
                "Person in proximity without proper PPE",
                time.time() - self.start_time
            )

        return None

    def get_recent_violations(self, window_seconds=10.0):
        """Get violations from recent time window"""
        current_time = time.time() - self.start_time
        cutoff_time = current_time - window_seconds

        recent = [v for v in self.violations if v.timestamp >= cutoff_time]
        return recent

    def get_violation_summary(self):
        """Get summary of violations by type"""
        summary = {
            ViolationType.CRITICAL: 0,
            ViolationType.WARNING: 0,
            ViolationType.CAUTION: 0
        }

        for violation in self.violations:
            summary[violation.type] += 1

        return summary

    def get_current_status(self):
        """
        Get current compliance status
        Returns worst violation type from recent window
        """
        recent = self.get_recent_violations(window_seconds=5.0)

        if not recent:
            return ViolationType.NONE

        # Return worst violation
        for v_type in [ViolationType.CRITICAL, ViolationType.WARNING, ViolationType.CAUTION]:
            if any(v.type == v_type for v in recent):
                return v_type

        return ViolationType.NONE

    def reset(self):
        """Reset monitor state"""
        self.frame_buffer.clear()
        self.violations.clear()
        self.last_sanitizer_time = None
        self.start_time = time.time()


def test_monitor():
    """Test monitor with dummy predictions"""
    import torch

    print("Testing ComplianceMonitor...")

    monitor = ComplianceMonitor(buffer_size=30, sanitizer_timeout=60.0)

    # Simulate 100 frames
    for i in range(100):
        # Create dummy predictions
        predictions = {
            'detection': torch.randn(1, 27, 60, 80),
            'segmentation': torch.randn(1, 8, 480, 640),
            'depth': torch.randn(1, 1, 480, 640),
            'keypoints': torch.randn(1, 21, 480, 640)
        }

        violations = monitor.update(predictions)

        if violations:
            print(f"Frame {i}: {len(violations)} violations")
            for v in violations:
                print(f"  {v}")

        # Simulate 0.1s delay between frames
        time.sleep(0.01)

    # Print summary
    print("\nViolation Summary:")
    summary = monitor.get_violation_summary()
    for v_type, count in summary.items():
        print(f"  {v_type.value}: {count}")

    print(f"\nCurrent Status: {monitor.get_current_status().value}")

    # Check buffer
    print(f"Buffer size: {len(monitor.frame_buffer)}/{monitor.buffer_size}")

    print("\nMonitor test completed!")


if __name__ == "__main__":
    test_monitor()
