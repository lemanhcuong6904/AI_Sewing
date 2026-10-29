#!/usr/bin/env python3
"""
Jetson Python script to communicate with STM32 motor controller via UART.
Dependencies: pip install pyserial
"""

from typing import Optional
import argparse
import time
import sys
import serial
import serial.tools.list_ports


def get_default_port() -> str:
    """Detect USB serial devices first (fallback to /dev/ttyUSB0)."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.description or "ACM" in p.device:
            return p.device
    return "/dev/ttyUSB0"


def send_and_receive(
    ser: Optional[serial.Serial],
    command: str,
    timeout: float = 1.0,
    wait_response: bool = False,
) -> str:
    """Send a UART command to STM32 and optionally wait for response.

    Args:
        ser: Open serial.Serial instance (or None for dry-run).
        command: Command string (newline appended automatically).
        timeout: Max seconds to wait for response if wait_response is True.
        wait_response: Whether to wait and read response from STM32.

    Returns:
        Response string (empty if wait_response is False or on timeout).
    """
    if ser is None:
        print(f"[dry-run] Would send: {command.strip()!r}")
        return ""

    if not command.endswith('\n'):
        command += '\n'

    ser.reset_input_buffer()
    print(f"--> TX: {command.strip()!r}")
    ser.write(command.encode('utf-8'))
    ser.flush()

    if not wait_response or timeout <= 0:
        return ""

    start_time = time.time()
    response = ""
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore')
            if line:
                print(f"<-- RX: {line.strip()}")
                response += line
                # if '\n' in line:
                #     break
        else:
            time.sleep(0.01)

    if response:
        print(f"<-- RX: {response.strip()}")
    else:
        print("<-- WARNING: No response from STM32 (timeout)")
    return response


# def interactive_mode(ser: serial.Serial):
#     print("\n=== CHẾ ĐỘ TƯƠNG TÁC (INTERACTIVE MODE) ===")
#     print(" - Nhập tọa độ dạng 'M1,M2' (ví dụ: 200,100) để chạy motor.")
#     print(" - Nhập 'v' hoặc 'V' để xem tọa độ mục tiêu hiện tại.")
#     print(" - Nhập 'exit' hoặc 'quit' để thoát.")
#     print("============================================\n")
#     
#     while True:
#         try:
#             cmd = input("Nhập lệnh: ").strip()
#             if not cmd:
#                 continue
#             if cmd.lower() in ['exit', 'quit']:
#                 break
#             send_and_receive(ser, cmd)
#         except KeyboardInterrupt:
#             print("\nĐã thoát.")
#             break
#         except Exception as e:
#             print(f"Lỗi: {e}")

def main():
    default_port = get_default_port()
    
    parser = argparse.ArgumentParser(description="Gửi lệnh UART từ NVIDIA Jetson tới STM32.")
    parser.add_argument("-p", "--port", default=default_port, 
                        help=f"Cổng Serial (Mặc định: {default_port}). Jetson Nano thường dùng /dev/ttyTHS1 (J41 Pin 8,10) hoặc /dev/ttyUSB0.")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="Tốc độ Baudrate (Mặc định: 115200)")
    parser.add_argument("--m1", type=float, help="Quãng đường M1 (mm)")
    parser.add_argument("--m2", type=float, help="Quãng đường M2 (mm)")
    parser.add_argument("-r", "--read", action="store_true", help="Gửi lệnh 'v' để đọc tọa độ hiện tại")
    parser.add_argument("-i", "--interactive", action="store_true", help="Chạy chế độ tương tác nhập tay")
    
    args = parser.parse_args()
    
    print(f"Đang kết nối tới {args.port} với Baudrate {args.baud}...")
    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0
        )
    except Exception as e:
        print(f"Không thể mở cổng {args.port}: {e}")
        print("\nGợi ý trên Jetson:")
        print("  1. Cấp quyền truy cập cổng: sudo chmod 666 <port>")
        print("  2. Kiểm tra dây nối TX-RX chéo nhau (TX Jetson -> RX STM32, RX Jetson -> TX STM32).")
        print("  3. Đối với Jetson J41 Header (Pin 8 - TXD, Pin 10 - RXD, Pin 9 - GND), cổng là /dev/ttyTHS1.")
        sys.exit(1)
        
    try:
        # Nếu có tham số m1 và m2
        if args.m1 is not None and args.m2 is not None:
            cmd = f"{args.m1},{args.m2}"
            send_and_receive(ser, cmd)
        # Nếu có lệnh đọc tọa độ
        elif args.read:
            send_and_receive(ser, "v")
        # Hoặc chạy chế độ tương tác nếu được yêu cầu hoặc không truyền tham số nào
        elif args.interactive or (args.m1 is None and args.m2 is None):
            interactive_mode(ser)
    finally:
        ser.close()
        print("Đã đóng kết nối Serial.")

if __name__ == "__main__":
    main()
