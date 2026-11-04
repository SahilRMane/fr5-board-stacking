"""Real model -> calibrated ROS vision -> world model, on saved images only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import rclpy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from hk_fr5_task_interfaces.msg import World as WorldMsg
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter,ParameterValue,ParameterType

ROOT = Path(__file__).resolve().parents[1]
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ.setdefault('ROS_DOMAIN_ID','169')
rclpy.init()
node = rclpy.create_node('dual_perception_integration_check')
packets,worlds = [],[]
sub = node.create_subscription(String,'/vision/boards',lambda m:packets.append(json.loads(m.data)),10)
world_sub = node.create_subscription(WorldMsg,'/task/world',worlds.append,10)
client = node.create_client(Trigger,'/vision/detect')
processes = []
def until(predicate, seconds=30, allow_exit=False):
    deadline = time.monotonic()+seconds
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.05)
        if predicate(): return True
        if not allow_exit:
            assert all(p.poll() is None for p in processes), 'Vision/world process exited; inspect test logs'
    return bool(predicate())
def stop():
    for p in reversed(processes):
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
    processes.clear()

try:
    with tempfile.TemporaryDirectory(prefix='dual model ä ') as directory:
        folder = Path(directory)
        corrupt = folder/'broken image.jpg';corrupt.write_text('not an image')
        for path,reason in ((folder/'missing image.jpg','Replay image missing'),(corrupt,'cannot be decoded')):
            failed = subprocess.run([sys.executable,'vision1/node.py','--ros-args','-p',f'device:={path}'],
                                    cwd=ROOT,capture_output=True,text=True,timeout=20)
            assert failed.returncode != 0 and reason in failed.stderr
            assert node.count_publishers('/vision/boards') == 0
        for backend in ('yolo26','yolo11'):
            packets.clear();worlds.clear()
            image = folder/'input.jpg'
            image.write_bytes((ROOT/'samples/01.jpg').read_bytes())
            logs = [(folder/(backend+'-'+name+'.log')).open('w') for name in ('vision','world')]
            try:
                for command,log in zip((
                    [sys.executable,'vision1/node.py','--ros-args','-p','model:=true','-p',f'backend:={backend}',
                     '-p',f'device:={image}','-p','rate:=3.0'],
                    ['ros2','run','hk_fr5_task','world_model']),logs):
                    processes.append(subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                                       start_new_session=True))
                assert until(lambda: packets and packets[-1]['boards'] and worlds and worlds[-1].boards,60), backend
                packet = packets[-1]
                assert packet['perception_backend']==backend and packet['calibrated']
                assert all(b['variant']!='hand' and np.isfinite(b['corners_xy']).all() for b in packet['boards'])
                assert client.wait_for_service(timeout_sec=10)
                requested = time.time()
                future = client.call_async(Trigger.Request())
                assert until(future.done,15) and future.result().success
                assert json.loads(future.result().message)['stamp']>=requested
                assert node.count_publishers('/vision/boards')==1
                parameters = node.create_client(SetParameters,'/vision1_overhead/set_parameters')
                assert parameters.wait_for_service(timeout_sec=10)
                changed_backend = Parameter(name='backend',value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,string_value='yolo11' if backend=='yolo26' else 'yolo26'))
                future = parameters.call_async(SetParameters.Request(parameters=[changed_backend]))
                assert until(future.done,10) and not future.result().results[0].successful, 'Live model switch was allowed'
                duplicate = subprocess.run([sys.executable,'vision1/node.py','--ros-args','-p','model:=true',
                    '-p',f'backend:={backend}','-p',f'device:={image}'],cwd=ROOT,capture_output=True,text=True,timeout=30)
                assert duplicate.returncode != 0 and 'another node already publishes' in duplicate.stderr
                assert node.count_publishers('/vision/boards')==1, 'Duplicate source disturbed active model'
                assert not any(name in ('motion_controller','gripper_controller')
                               for name,_ in node.get_node_names_and_namespaces())
                assert cv2.imwrite(str(image),np.zeros((1080,1920,3),np.uint8))
                changed = time.time()
                assert until(lambda: packets[-1]['stamp']>=changed and not packets[-1]['boards'],15)
                assert until(lambda: not worlds[-1].boards,15), 'World retained removed boards'
                assert cv2.imwrite(str(image),np.zeros((480,640,3),np.uint8))
                future = client.call_async(Trigger.Request())
                assert until(future.done,15) and not future.result().success, 'Wrong-resolution image used with calibration'
                image.unlink()
                future = client.call_async(Trigger.Request())
                assert until(future.done,15) and not future.result().success, 'Missing image served cached detections'
                image.write_bytes((ROOT/'samples/01.jpg').read_bytes())
                assert until(lambda: packets[-1]['boards'] and worlds[-1].vision_ok and worlds[-1].boards,15)
                processes[0].kill();processes[0].wait(timeout=5)
                assert until(lambda: not worlds[-1].vision_ok,15,allow_exit=True), 'Vision crash left the world marked fresh'
                print(f'PASS {backend}: calibrated detections, world tracking/removal, snapshots, source loss/crash and switch rejection',flush=True)
            finally:
                stop()
                for log in logs:log.close()
                # Let DDS discard the old publisher before selecting the next model.
                assert until(lambda: node.count_publishers('/vision/boards')==0,10)
        print('2/2 real-model ROS pipelines passed (no hardware)',flush=True)
except Exception:
    for log in logs:
        log.flush() if not log.closed else None
    for path in folder.glob('*.log'):
        print(path.name, path.read_text()[-3000:],file=sys.stderr)
    raise
finally:
    stop();node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
