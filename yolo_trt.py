from __future__ import annotations
import threading
from pathlib import Path
import numpy as np

COCO = [
'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
'hair drier','toothbrush']

class TRTDetector:
    def __init__(self, engine_path: str):
        self.engine_path = str(engine_path)
        self.lock = threading.Lock()
        self.ready = False
        self.error = ''
        self._load()

    def _load(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            cuda.init()
            self.trt, self.cuda = trt, cuda
            self.cuda_ctx = cuda.Device(0).retain_primary_context()
            self.cuda_ctx.push()
            try:
                logger = trt.Logger(trt.Logger.ERROR)
                data = Path(self.engine_path).read_bytes()
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(data)
                if self.engine is None:
                    raise RuntimeError('TensorRT could not deserialize engine')
                self.context = self.engine.create_execution_context()
                self.stream = cuda.Stream()
                names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
                self.input_name = next((n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT), None)
                self.output_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
                if not self.input_name:
                    raise RuntimeError('No TensorRT input tensor found')
                shape = tuple(int(x) for x in self.engine.get_tensor_shape(self.input_name))
                if any(x < 0 for x in shape):
                    shape = (1,3,1088,1088)
                    self.context.set_input_shape(self.input_name, shape)
                self.input_shape = shape
                self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
                self.d_input = cuda.mem_alloc(int(np.prod(shape) * self.input_dtype.itemsize))
                self.context.set_tensor_address(self.input_name, int(self.d_input))
                self.outputs = []
                for name in self.output_names:
                    oshape = tuple(int(x) for x in self.context.get_tensor_shape(name))
                    if any(x < 0 for x in oshape):
                        oshape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
                    dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                    host = cuda.pagelocked_empty(int(np.prod(oshape)), dtype)
                    dev = cuda.mem_alloc(host.nbytes)
                    self.context.set_tensor_address(name, int(dev))
                    self.outputs.append((name, oshape, host, dev))
                self.ready = True
            finally:
                self.cuda_ctx.pop()
        except Exception as e:
            self.error = f'{type(e).__name__}: {e}'
            self.ready = False

    def info(self):
        return {'ready':self.ready,'error':self.error,'engine':self.engine_path,
                'input_shape':getattr(self,'input_shape',None),
                'outputs':[{'name':x[0],'shape':x[1]} for x in getattr(self,'outputs',[])]}

    def _preprocess(self, frame):
        import cv2
        _,_,nh,nw = self.input_shape
        resized = cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        arr = np.transpose(rgb.astype(np.float32)/255.0,(2,0,1))[None,...]
        return np.ascontiguousarray(arr.astype(self.input_dtype)), (frame.shape[1],frame.shape[0]), (nw,nh)

    def infer(self, frame, conf_thres=0.35, iou_thres=0.45):
        if not self.ready: return []
        import cv2
        with self.lock:
            self.cuda_ctx.push()
            try:
                inp, orig, net = self._preprocess(frame)
                self.cuda.memcpy_htod_async(self.d_input,inp,self.stream)
                if not self.context.execute_async_v3(stream_handle=self.stream.handle):
                    return []
                for _,_,host,dev in self.outputs:
                    self.cuda.memcpy_dtoh_async(host,dev,self.stream)
                self.stream.synchronize()
                arrays=[host.reshape(shape).copy() for _,shape,host,_ in self.outputs]
            finally:
                self.cuda_ctx.pop()
        if not arrays: return []
        out=max(arrays,key=lambda a:a.size)
        out=np.squeeze(out)
        ow,oh=orig; nw,nh=net; sx,sy=ow/nw,oh/nh
        detections=[]
        if out.ndim==2 and out.shape[-1]==6:
            for r in out:
                x1,y1,x2,y2,score,cls=[float(x) for x in r]
                if score < conf_thres: continue
                ci=int(cls)
                detections.append({'bbox':[int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)],'conf':score,
                                   'class_id':ci,'label':COCO[ci] if 0<=ci<len(COCO) else str(ci)})
            return detections
        if out.ndim != 2: return []
        if out.shape[0] in (84,85) or (out.shape[0] < out.shape[1] and out.shape[0] < 128): out=out.T
        if out.shape[1] < 6: return []
        boxes=[]; scores=[]; class_ids=[]
        for r in out:
            cls_scores=r[4:]
            ci=int(np.argmax(cls_scores)); score=float(cls_scores[ci])
            if score < conf_thres: continue
            cx,cy,bw,bh=[float(x) for x in r[:4]]
            x1=max(0,min(ow-1,(cx-bw/2)*sx)); y1=max(0,min(oh-1,(cy-bh/2)*sy))
            x2=max(0,min(ow-1,(cx+bw/2)*sx)); y2=max(0,min(oh-1,(cy+bh/2)*sy))
            boxes.append([int(x1),int(y1),int(x2-x1),int(y2-y1)]); scores.append(score); class_ids.append(ci)
        if not boxes: return []
        idxs=cv2.dnn.NMSBoxes(boxes,scores,conf_thres,iou_thres)
        if len(idxs)==0: return []
        for i in np.array(idxs).reshape(-1):
            i=int(i); x,y,w,h=boxes[i]; ci=class_ids[i]
            detections.append({'bbox':[x,y,x+w,y+h],'conf':float(scores[i]),'class_id':ci,
                               'label':COCO[ci] if 0<=ci<len(COCO) else str(ci)})
        return detections
