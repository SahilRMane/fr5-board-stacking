"""Both trained models feed the same calibrated board contract; no ROS or hardware."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vision1'))
import seg
import detect
import table
from hk_fr5_task import vision1
from hk_fr5_task.world_core import World


class PerceptionModelsCheck(unittest.TestCase):
    def tearDown(self):
        seg.configure('yolo26')

    def test_explicit_selection_and_no_silent_fallback(self):
        for selection in ('automatic','YOLO11',None,False,[],{}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                seg.configure(selection)
        for backend in ('yolo26', 'yolo11'):
            seg.configure(backend)
            with patch.object(seg, 'BACKENDS', {backend:dict(seg.settings(), path='models/missing.onnx')}):
                with self.assertRaises(FileNotFoundError):
                    detect.detect(np.zeros((1080,1920,3), np.uint8), use_model=True)

    def test_corrupt_model_refused_before_runtime_load(self):
        with patch.object(seg, 'BACKENDS', {'yolo26':dict(seg.settings(), sha256='0'*64)}):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                seg._session()

    def test_bundled_models_have_valid_metadata_and_checksum(self):
        for backend in seg.BACKENDS:
            with self.subTest(backend=backend):
                seg.configure(backend)
                self.assertTrue(seg.available())
                self.assertEqual(hashlib.sha256((ROOT/seg.settings()['path']).read_bytes()).hexdigest(),
                                 seg.settings()['sha256'])
                self.assertEqual(seg._session().get_inputs()[0].shape, [1,3,960,960])
                import onnxruntime as ort
                seg.configure(backend)
                with patch.object(ort,'get_available_providers',return_value=['CPUExecutionProvider']):
                    self.assertEqual(seg._session().get_providers(),['CPUExecutionProvider'])

    def test_hand_class_cannot_be_published_as_a_board(self):
        seg.configure('yolo11')
        class FakeSession:
            def get_inputs(self):
                return [type('Input', (), dict(shape=[1,3,960,960], name='images'))()]
            def run(self, *_):
                predictions = np.zeros((1,40,1),np.float32)
                predictions[0,:4,0] = [480,480,200,100]
                predictions[0,7,0] = .99
                return predictions, np.ones((1,32,240,240),np.float32)
        with patch.object(seg, '_session', return_value=FakeSession()):
            self.assertEqual(seg.segment(np.zeros((1080,1920,3),np.uint8),(0,0,1920,1080)), [])

    def test_invalid_images_and_rois_fail_before_inference(self):
        image = np.zeros((100,200,3), np.uint8)
        for invalid,roi in ((image.astype(float),(0,0,200,100)), (image,(-1,0,200,100)),
                            (image,(0,0,201,100)),(image,(0,0,0,100))):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                seg.segment(invalid, roi)
        for bad in (None,[],image[:,:,0],np.empty((0,200,3),np.uint8),image.astype(np.float32)):
            with self.subTest(image=type(bad)),self.assertRaises(ValueError):
                detect.detect(bad,use_model=True)
        for roi in (None,[],[0,0,200],[0.,0,200,100],[False,0,200,100]):
            with self.subTest(roi=roi),self.assertRaises(ValueError):
                seg.segment(image,roi)

    def test_model_filenames_with_spaces_and_unicode(self):
        seg.configure('yolo11')
        cfg = seg.settings()
        with tempfile.TemporaryDirectory(prefix='model path ') as directory:
            path = Path(directory)/'board model ä.onnx'
            path.write_bytes((ROOT/cfg['path']).read_bytes())
            with patch.object(seg,'BACKENDS',{'yolo11':dict(cfg,path=str(path))}):
                self.assertEqual(seg._session().get_inputs()[0].shape,[1,3,960,960])

    def test_malformed_output_is_refused_and_empty_output_is_valid(self):
        seg.configure('yolo11')
        class FakeSession:
            def get_inputs(self):
                return [type('Input',(),dict(shape=[1,3,960,960],name='images'))()]
            def run(self,*_):return self.outputs
        session = FakeSession()
        valid = np.zeros((1,40,1),np.float32)
        valid[0,:4,0] = [480,480,200,100];valid[0,4,0]=.9
        masks = np.ones((1,32,240,240),np.float32)
        bad_outputs = [(valid[0],masks),(valid,masks[0]),(valid,np.zeros((1,31,240,240),np.float32)),
                       (valid.astype(object),masks),(valid,np.full_like(masks,np.nan)),
                       (np.full_like(valid,np.inf),masks)]
        for index,value in ((4,1.2),(4,-.1),(2,-1),(3,0)):
            bad = valid.copy();bad[0,index,0]=value;bad_outputs.append((bad,masks))
        with patch.object(seg,'_session',return_value=session):
            for output in bad_outputs:
                session.outputs = output
                with self.subTest(shape=output[0].shape),self.assertRaises(ValueError):
                    seg.segment(np.zeros((1080,1920,3),np.uint8),(0,0,1920,1080))
            session.outputs = (np.empty((1,40,0),np.float32),masks)
            self.assertEqual(seg.segment(np.zeros((1080,1920,3),np.uint8),(0,0,1920,1080)),[])

    def test_bad_manifest_values_do_not_select_a_model(self):
        cfg = seg.settings()
        for key,value in (('confidence','0.4'),('confidence',float('nan')),('confidence',True),
                          ('full_frame','false'),('classes',['hand','VARa','VARb','VARc'])):
            with patch.object(seg,'BACKENDS',{'yolo26':dict(cfg,**{key:value})}):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    seg.configure('yolo26')

    def test_invalid_calibration_and_numeric_conversion(self):
        raw = yaml.safe_load((ROOT/'config/calib.yaml').read_text())
        bad_cases = []
        for key,value in (('image_size',[1920.,1080]),('homography',[[0]*3]*3),
                          ('homography',[[float('nan')]*3]*3),('lens_k','invalid'),('tags',[])):
            bad_cases.append(dict(raw,**{key:value}))
        bad_cases += [None,dict(raw,calibrated='true'),dict(raw,tags=[{'u':1,'v':1}]*4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration ä.yaml'
            for bad in bad_cases:
                path.write_text(yaml.safe_dump(bad))
                with self.subTest(calibration=bad),self.assertRaises(ValueError):table.load(path)
            raw['homography'] = [[str(v) for v in row] for row in raw['homography']]
            raw['lens_k'] = str(raw['lens_k'])
            path.write_text(yaml.safe_dump(raw))
            cfg = table.load(path)
            self.assertTrue(np.isfinite(table.to_base(cfg,[[1000,500]])).all())
            for pixels in ([[float('nan'),0]],[1,2],[[1,2,3]],[['bad',0]]):
                with self.subTest(pixels=pixels),self.assertRaises(ValueError):table.to_base(cfg,pixels)
            cfg['homography'] = [[1,0,0],[0,1,0],[0,0,0]]
            with self.assertRaisesRegex(ValueError,'singular'):table.to_base(cfg,[[1000,500]])

    def test_world_adapter_refuses_invalid_frames_and_timestamps(self):
        base = dict(stamp=10.,calibrated=True,boards=[],zones=[])
        for field,value in (('stamp',float('nan')),('stamp',float('inf')),('stamp',True),
                            ('stamp','invalid'),('calibrated','false'),('boards',{}),('zones',True),('obstacles','hand')):
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                vision1.parse(dict(base,**{field:value}))
        self.assertEqual(vision1.parse(dict(base,stamp='10.5'))[0],10.5)

    def test_bad_board_data_cannot_enter_world_model(self):
        board = dict(variant='VARa',conf=.9,inside_calibration=True,corners_xy=[[0,0],[1,0],[1,1],[0,1]])
        bad_cases = [None,0,dict(board,corners_xy=[[0,0]]*4),dict(board,corners_xy=[[0,0],[1,1],[1,0],[0,1]])]
        for key,value in (('conf',float('nan')),('conf',-1),('conf',2),('conf',True),
                          ('variant',None),('inside_calibration','false'),('inside_calibration',0),
                          ('corners_xy',[[float('nan'),0],[1,0],[1,1],[0,1]])):
            bad_cases.append(dict(board,**{key:value}))
        for bad in bad_cases:
            with self.subTest(board=bad):
                _,dets,problems = vision1.parse(dict(stamp=10.,calibrated=True,boards=[bad]))
                self.assertEqual(dets,[]);self.assertEqual(len(problems),1)

    def test_invalid_zones_and_obstacles_are_skipped_with_reasons(self):
        zone = dict(name='drop_off_a',kind='drop_off',x=0,y=0,yaw=0,size=[.3,.2],top_z=-.055)
        for bad in (None,[],dict(zone,size=[0,.2]),dict(zone,size=[.2,.3]),dict(zone,x=float('inf')),
                    dict(zone,name=None),dict(zone,top_z=True)):
            with self.subTest(zone=bad):
                _,dets,problems = vision1.parse(dict(stamp=10.,calibrated=True,zones=[bad],obstacles=[False]))
                self.assertEqual(dets,[]);self.assertEqual(len(problems),2)

    def test_runtime_failure_never_falls_back_to_classical_detection(self):
        import onnxruntime as ort
        seg.configure('yolo11')
        with patch.object(ort,'InferenceSession',side_effect=RuntimeError('injected runtime failure')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                detect.detect(np.zeros((1080,1920,3),np.uint8),use_model=True)
        self.assertIsNone(seg._sess)

    def test_yolo11_uses_one_full_frame_yolo26_keeps_two_crops(self):
        image = np.zeros((1080,1920,3),np.uint8)
        for backend,expected in (('yolo11',[(0,0,1920,1080)]),
                                 ('yolo26',[detect.roi_px(1920,1080,r) for r in detect.ROIS])):
            seg.configure(backend)
            with patch.object(seg, 'segment', return_value=[]) as run:
                detect.detect(image,use_model=True)
            self.assertEqual([c.args[1] for c in run.call_args_list],expected)

    def test_full_frame_masks_outside_workcell_are_rejected_without_clipping(self):
        seg.configure('yolo11')
        mask = np.zeros((1080,1920),np.uint8)
        mask[300:500,900:1200] = 255  # crosses the robot rail and table boundary
        rejected = []
        with patch.object(seg,'segment',return_value=[dict(cls='VARa',conf=.99,mask=mask)]):
            self.assertEqual(detect.detect(np.zeros((1080,1920,3),np.uint8),rejected,use_model=True),[])
        self.assertEqual(rejected[0]['reason'],'outside_roi')

    def test_both_models_feed_metric_adapter_and_world(self):
        calibration = table.load()
        config = yaml.safe_load((ROOT/'src/hk_fr5_task/config/world.yaml').read_text())
        for backend in seg.BACKENDS:
            seg.configure(backend)
            positives = 0
            for filename in ('01.jpg','02.jpg','03.jpg','06.jpg','07.jpg','08.jpg'):
                image = cv2.imread(str(ROOT/'samples'/filename))
                raw = detect.detect(image,use_model=True)
                self.assertTrue(all(b['model_class'] in ('VARa','VARb','VARc') for b in raw))
                rejected,occluded = [],[]
                boards = table.pickable(calibration,raw,rejected,occluded)
                for board in boards:
                    self.assertTrue(np.isfinite(board['corners_xy']).all())
                    self.assertLess(abs(board['yaw']),np.pi/2+1e-4)
                packet = dict(stamp=10.,calibrated=True,boards=boards,zones=[])
                stamp,detections,problems = vision1.parse(packet)
                self.assertEqual(problems,[])
                world = World(copy.deepcopy(config))
                for k in range(5):
                    world.update(stamp+k*.1,detections)
                self.assertIsInstance(world.snapshot(10.4)['boards'],list)
                positives += len(boards)
            self.assertGreater(positives,0,backend+' produced no calibrated boards on six lab images')


if __name__ == '__main__':
    unittest.main()
