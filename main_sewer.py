from typing import Tuple, Optional, List
import serial
import time
import sys
import tty
import termios
import argparse
import select

from utils.line_detectors.detector import DetectorMode

def parse_args():
    parser = argparse.ArgumentParser(description="AI Sewing Controller")
    parser.add_argument("--visualize", action="store_true", help="Visualize the image stream.")
    parser.add_argument("--interval", type=int, nargs=2, default=(-14, 15), help="Interval argument")
    parser.add_argument("--img-size", type=int, nargs=2, default=(128, 128), dest="img_size", help="Image size argument")
    parser.add_argument("--hist-thres", type=float, default=100000.0, dest="hist_threshold", help="Color histogram threshold argument")
    parser.add_argument("--dist-method", type=str, default='L2', dest="dist_method", help="Distance metric argument")
    parser.add_argument("--detector", type=str, default='lsd', choices=['lsd', 'fld', 'edlines', 'ensemble', 'mlsd'],
                        help="Line detector backend: lsd (default), fld, edlines, ensemble, or mlsd")
    parser.add_argument("--use-gray", action="store_true", help="Use gray image for histogram computation")
    return parser.parse_args()

from modules.fabric_sewer import run_cam_return_signal, run_cam_return_signal_with_img

# Config Serial port (Based on actual Jetson UART, eg: '/dev/ttyTHS1' or '/dev/ttyUSB0')
SERIAL_PORT_1 = '/dev/ttyUSB0' 
BAUD_RATE = 115200


def main(
   img_size: Tuple[int, int] = (112, 112),
   interval: Tuple[int, int] = (-3, 3),
   hist_threshold: float = 100000.0,
   dist_method: str = 'L2',
   visualize: bool = False,
   detector_mode: DetectorMode = DetectorMode.LSD,
   use_gray: bool = False,
   velocity: Optional[List] = None,
   step_overhead: float = 1.2,
):
    ser_1 = None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Open serial port
        ser_1 = serial.Serial(SERIAL_PORT_1, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for a stable connection
        print(f"Connected to {SERIAL_PORT_1} with baudrate {BAUD_RATE}.")
        print("\n--- ENGINE CONTROLLER INTERFACE ---")
        print("Type one of the keys below to control the speed of the engine:")
        print("  '1' - Slow")
        print("  '2' - Medium")
        print("  '3' - Fast")
        print("  'q' - Quit")
        print("-------------------------------------------\n")
        
        # Start the engine
        cmd_speed = '1'
        cmd_cam_1 = None
        cmd_cam_1_prev = None
        ser_1.write(cmd_speed.encode('utf-8'))

        # Shared mutable state for velocity-based error correction:
        #   speed_state[0]: current speed char ('1'/'2'/'3')
        #   speed_state[1]: True when rotor is in reverse mode ('b')
        if velocity is None:
            velocity = [2.5, 3.5, 7.0]
        speed_state = [cmd_speed, 1]  # [speed_char, direction: 1=fwd, 0=stopped, -1=rev]
        
        tty.setraw(fd)

        # Stream of frame from camera
        cam_func = run_cam_return_signal_with_img if visualize else run_cam_return_signal
        for control_signal_1, control_signal_2 in cam_func(
                interval=interval, img_size=img_size, hist_threshold=hist_threshold,
                dist_method=dist_method, detector_mode=detector_mode, use_gray=use_gray,
                velocity=velocity, step_overhead=step_overhead, speed_state=speed_state):

            # Firstly, check for a speed configuration key (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                cmd_speed = sys.stdin.read(1).strip().lower()
                if cmd_speed == 'q':
                    ser_1.write('0'.encode('utf-8'))
                    ser_1.flush()
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    ser_1.close()
                    print("Closed serial port 1.")
                    print("Exiting...")
                    break
                elif cmd_speed in ['1', '2', '3']:
                    # Send character to UART
                    ser_1.write(cmd_speed.encode('utf-8'))
                    ser_1.flush()
                    print(f"Sent speed: {cmd_speed}")
                    speed_state[0] = cmd_speed  # update shared speed level
            
            # Line 1 control logic
            if control_signal_1 is not None:
                if control_signal_1 == [False, False]: # Keep rolling
                    cmd_cam_1 = cmd_speed
                    speed_state[1] = 1   # forward
                elif control_signal_1 == [True, False]: # Stop the first motor
                    cmd_cam_1 = 'a'
                    speed_state[1] = 0   # stopped
                elif control_signal_1 == [True, True]: # Reverse motor
                    cmd_cam_1 = 'b'
                    speed_state[1] = -1  # reverse

            # Send character to UART
            if cmd_cam_1 is not None and cmd_cam_1 != cmd_cam_1_prev:
                ser_1.write(cmd_cam_1.encode('utf-8'))
                ser_1.flush()
                print(f"Sent camera 1: {cmd_cam_1}")
                cmd_cam_1_prev = cmd_cam_1

    except serial.SerialException as e:
        print(f"Error when opening serial port: {e}")
    except KeyboardInterrupt:
        print("\n Exiting...")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if ser_1 is not None and ser_1.is_open:
            ser_1.close()
            print("Closed serial port 1.")


if __name__ == '__main__':
    args = parse_args()
    mode = DetectorMode(args.detector)
    main(interval=tuple(args.interval), img_size=tuple(args.img_size), hist_threshold=args.hist_threshold,
         visualize=args.visualize, dist_method=args.dist_method, detector_mode=mode, use_gray=args.use_gray)