import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import {
  Upload,
  Camera,
  Zap,
  FileText,
  Folder,
  Image,
  Video,
  Download,
  ArrowLeft,
  X,
  Layers,
  Activity,
  Maximize,
  Play,
  Pause,
  Cpu,
  Clock,
  RefreshCw,
  ChevronDown,
  CheckCircle,
  AlertTriangle,
  Info,
  StopCircle,
  Lock,
  Loader2,
  FlipHorizontal
} from 'lucide-react';

// 导入推理服务和权重树选择器
import { inferenceService, WeightLibrary, InferencePredictResponse, InferenceResult, WeightTreeSelectOption } from '../../services/inference';
import WeightTreeSelect from '../shared/WeightTreeSelect';

// ==================== 持久化存储工具 ====================

// IndexedDB 操作封装
const STORAGE_DB_NAME = 'CVStudioInference';
const STORAGE_STORE_NAME = 'inferenceData';

const getDB = () => {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(STORAGE_DB_NAME, 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORAGE_STORE_NAME)) {
        db.createObjectStore(STORAGE_STORE_NAME);
      }
    };
  });
};

const saveFile = async (key: string, file: File): Promise<void> => {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORAGE_STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORAGE_STORE_NAME);
    store.put(file, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

const deleteFile = async (key: string): Promise<void> => {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORAGE_STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORAGE_STORE_NAME);
    store.delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

const loadFile = async (key: string): Promise<File | null> => {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORAGE_STORE_NAME, 'readonly');
    const store = tx.objectStore(STORAGE_STORE_NAME);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
};

// LocalStorage 键名
const LS_KEYS = {
  MODE: 'inference_mode',
  SINGLE_IMAGE: 'inference_single_image',
  SINGLE_RESULT: 'inference_single_result',
  BATCH_IMAGES: 'inference_batch_images',
  BATCH_RESULTS: 'inference_batch_results',
  BATCH_STATUS: 'inference_batch_status',
  INFERENCE_STATUS: 'inference_status',
  SELECTED_WEIGHT: 'inference_selected_weight'
};

const InferenceView: React.FC = () => {
  const [mode, setMode] = useState<'single' | 'batch' | 'stream'>('single');
  const [selectedBatchImage, setSelectedBatchImage] = useState<number | null>(null);
  const [selectedWeightId, setSelectedWeightId] = useState<number | null>(null);

  // 权重树和加载状态
  const [weightTree, setWeightTree] = useState<WeightTreeSelectOption[]>([]);
  const [weightsLoading, setWeightsLoading] = useState(true);

  // 推理状态
  const [inferenceStatus, setInferenceStatus] = useState<'idle' | 'processing' | 'completed' | 'error'>('idle');
  const [inferenceResult, setInferenceResult] = useState<InferencePredictResponse | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // 图像尺寸（用于检测框坐标计算）
  const [imageDimensions, setImageDimensions] = useState<{naturalWidth: number, naturalHeight: number} | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // 置信度阈值
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);

  // Stream Mode State
  const [streamStatus, setStreamStatus] = useState<'idle' | 'scanning' | 'ready' | 'live'>('idle');
  const [cameras, setCameras] = useState<{id: string, label: string}[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [videoStreamRef, setVideoStreamRef] = useState<MediaStream | null>(null);
  const [isMirrored, setIsMirrored] = useState<boolean>(true); // 默认镜像

  // Batch Mode State - 批量图片管理
  const [batchImages, setBatchImages] = useState<{file: File, preview: string, name: string}[]>([]);
  const [batchInferenceStatus, setBatchInferenceStatus] = useState<'idle' | 'processing' | 'completed'>('idle');
  const [batchResults, setBatchResults] = useState<Map<number, InferencePredictResponse>>(new Map());
  const batchInputRef = useRef<HTMLInputElement>(null);

  // 文件上传引用
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 视频元素引用
  const videoRef = useRef<HTMLVideoElement>(null);

  // Notification State
  const [notification, setNotification] = useState<{msg: string, type: 'error' | 'success' | 'info'} | null>(null);
  const [isRestoring, setIsRestoring] = useState(true); // 恢复状态标志

  const showNotification = (msg: string, type: 'error' | 'success' | 'info') => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // ==================== 持久化函数 ====================

  // 保存单图模式状态
  const saveSingleState = useCallback(async () => {
    if (uploadedFile) {
      await saveFile('single_image', uploadedFile);
      localStorage.setItem(LS_KEYS.SINGLE_IMAGE, 'saved');
    } else {
      localStorage.removeItem(LS_KEYS.SINGLE_IMAGE);
    }
    if (inferenceResult) {
      localStorage.setItem(LS_KEYS.SINGLE_RESULT, JSON.stringify(inferenceResult));
    } else {
      localStorage.removeItem(LS_KEYS.SINGLE_RESULT);
    }
    localStorage.setItem(LS_KEYS.INFERENCE_STATUS, inferenceStatus);
  }, [uploadedFile, inferenceResult, inferenceStatus]);

  // 保存批量模式状态
  const saveBatchState = useCallback(async () => {
    // 保存批量图片文件列表的元数据
    const batchMetadata = batchImages.map((img, idx) => ({
      idx,
      name: img.name,
      hasResult: batchResults.has(idx)
    }));
    localStorage.setItem(LS_KEYS.BATCH_IMAGES, JSON.stringify(batchMetadata));

    // 保存每个图片文件
    for (let i = 0; i < batchImages.length; i++) {
      await saveFile(`batch_image_${i}`, batchImages[i].file);
    }

    // 保存批量结果
    const resultsData: Record<number, any> = {};
    batchResults.forEach((value, key) => {
      resultsData[key] = value;
    });
    localStorage.setItem(LS_KEYS.BATCH_RESULTS, JSON.stringify(resultsData));
    localStorage.setItem(LS_KEYS.BATCH_STATUS, batchInferenceStatus);
  }, [batchImages, batchResults, batchInferenceStatus]);

  // 恢复单图模式状态
  const restoreSingleState = useCallback(async () => {
    const hasImage = localStorage.getItem(LS_KEYS.SINGLE_IMAGE);
    const resultStr = localStorage.getItem(LS_KEYS.SINGLE_RESULT);
    const status = localStorage.getItem(LS_KEYS.INFERENCE_STATUS) as any;

    if (hasImage === 'saved') {
      const file = await loadFile('single_image');
      if (file) {
        const url = URL.createObjectURL(file);
        setUploadedImageUrl(url);
        setUploadedFile(file);
      }
    }

    if (resultStr) {
      try {
        setInferenceResult(JSON.parse(resultStr));
      } catch (e) {
        console.error('恢复单图推理结果失败:', e);
      }
    }

    if (status) {
      // 如果保存的状态是 processing，说明页面刷新时推理被打断了
      // 重置为 idle 状态，避免界面卡在"推理中"
      if (status === 'processing') {
        setInferenceStatus('idle');
        setErrorMessage(null);
      } else {
        setInferenceStatus(status);
      }
    }
  }, []);

  // 恢复批量模式状态
  const restoreBatchState = useCallback(async () => {
    const metadataStr = localStorage.getItem(LS_KEYS.BATCH_IMAGES);
    const resultsStr = localStorage.getItem(LS_KEYS.BATCH_RESULTS);
    const status = localStorage.getItem(LS_KEYS.BATCH_STATUS) as any;

    if (metadataStr) {
      try {
        const metadata = JSON.parse(metadataStr);
        const images: {file: File, preview: string, name: string}[] = [];

        for (const meta of metadata) {
          const file = await loadFile(`batch_image_${meta.idx}`);
          if (file) {
            images.push({
              file,
              preview: URL.createObjectURL(file),
              name: meta.name
            });
          }
        }

        if (images.length > 0) {
          setBatchImages(images);
        }
      } catch (e) {
        console.error('恢复批量图片失败:', e);
      }
    }

    if (resultsStr) {
      try {
        const resultsData = JSON.parse(resultsStr);
        const results = new Map<number, InferencePredictResponse>();
        Object.entries(resultsData).forEach(([key, value]) => {
          results.set(parseInt(key), value as InferencePredictResponse);
        });
        setBatchResults(results);
      } catch (e) {
        console.error('恢复批量结果失败:', e);
      }
    }

    if (status) {
      // 如果保存的状态是 processing，说明页面刷新时推理被打断了
      // 重置为 idle 状态，避免界面卡在"推理中"
      if (status === 'processing') {
        setBatchInferenceStatus('idle');
      } else {
        setBatchInferenceStatus(status);
      }
    }
  }, []);

  // 清除当前模式的状态
  const clearCurrentModeState = useCallback(async () => {
    if (mode === 'single') {
      localStorage.removeItem(LS_KEYS.SINGLE_IMAGE);
      localStorage.removeItem(LS_KEYS.SINGLE_RESULT);
      localStorage.removeItem(LS_KEYS.INFERENCE_STATUS);
      await deleteFile('single_image'); // 删除 IndexedDB 中的文件
    } else if (mode === 'batch') {
      localStorage.removeItem(LS_KEYS.BATCH_IMAGES);
      localStorage.removeItem(LS_KEYS.BATCH_RESULTS);
      localStorage.removeItem(LS_KEYS.BATCH_STATUS);
      // 清除批量图片文件
      for (let i = 0; i < batchImages.length; i++) {
        await deleteFile(`batch_image_${i}`);
      }
    }
  }, [mode, batchImages.length]);

  // 加载权重树
  useEffect(() => {
    const loadWeightTree = async () => {
      setWeightsLoading(true);
      try {
        const tree = await inferenceService.getWeightTree();
        setWeightTree(tree);
        // 恢复之前选择的权重
        const savedWeightId = localStorage.getItem(LS_KEYS.SELECTED_WEIGHT);
        if (savedWeightId) {
          const num = parseInt(savedWeightId);
          if (findWeightInTree(tree, num)) {
            setSelectedWeightId(num);
          } else if (tree.length > 0) {
            setSelectedWeightId(tree[0].id);
          }
        } else if (tree.length > 0) {
          setSelectedWeightId(tree[0].id);
        }
      } catch (error) {
        console.error('加载权重树失败:', error);
        showNotification("加载权重列表失败", "error");
      } finally {
        setWeightsLoading(false);
      }
    };
    loadWeightTree();
  }, []);

  // 辅助函数：在权重树中查找权重
  const findWeightInTree = (tree: WeightTreeSelectOption[], id: number): WeightTreeSelectOption | null => {
    for (const weight of tree) {
      if (weight.id === id) return weight;
      if (weight.children) {
        const found = findWeightInTree(weight.children, id);
        if (found) return found;
      }
    }
    return null;
  };

  // 获取选中的权重信息
  const selectedWeight = useMemo(() => {
    if (!selectedWeightId) return null;
    return findWeightInTree(weightTree, selectedWeightId);
  }, [weightTree, selectedWeightId]);

  // 恢复保存的模式和状态
  useEffect(() => {
    const restoreState = async () => {
      const savedMode = localStorage.getItem(LS_KEYS.MODE) as 'single' | 'batch' | 'stream' | null;
      if (savedMode && savedMode !== 'stream') { // 不恢复视频流模式
        setMode(savedMode);
        if (savedMode === 'single') {
          await restoreSingleState();
        } else if (savedMode === 'batch') {
          await restoreBatchState();
        }
      }
      setIsRestoring(false);
    };
    restoreState();
  }, [restoreSingleState, restoreBatchState]);

  // 保存模式变化
  useEffect(() => {
    if (!isRestoring) {
      localStorage.setItem(LS_KEYS.MODE, mode);
    }
  }, [mode, isRestoring]);

  // Check if controls should be locked
  const isLocked = streamStatus === 'live';

  // Derive task type from selected weight
  // 获取实际的任务类型（从推理结果中获取）
  const actualTaskType: 'classification' | 'detection' = useMemo(() => {
    return inferenceResult?.task_type || 'detection';
  }, [inferenceResult]);

  // Reset stream state when switching modes
  useEffect(() => {
      if (mode !== 'stream') {
          // 停止视频流
          if (videoStreamRef) {
              videoStreamRef.getTracks().forEach(track => track.stop());
              setVideoStreamRef(null);
          }
          setStreamStatus('idle');
          setCameras([]);
          setSelectedCamera('');
      }
      // 切换模式时不再清空批量数据，保留以实现持久化
  }, [mode]);

  // 保存权重选择
  useEffect(() => {
    if (selectedWeightId) {
      localStorage.setItem(LS_KEYS.SELECTED_WEIGHT, String(selectedWeightId));
    }
  }, [selectedWeightId]);

  // 自动保存单图模式状态
  useEffect(() => {
    if (!isRestoring && mode === 'single') {
      saveSingleState();
    }
  }, [uploadedFile, inferenceResult, inferenceStatus, isRestoring, mode, saveSingleState]);

  // 自动保存批量模式状态
  useEffect(() => {
    if (!isRestoring && mode === 'batch') {
      saveBatchState();
    }
  }, [batchImages, batchResults, batchInferenceStatus, isRestoring, mode, saveBatchState]);

  // 组件卸载时清理资源
  useEffect(() => {
      return () => {
          if (videoStreamRef) {
              videoStreamRef.getTracks().forEach(track => track.stop());
          }
      };
  }, [videoStreamRef]);

  // --- Actions ---

  // 处理图片选择
  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // 验证文件类型
    if (!file.type.startsWith('image/')) {
      showNotification('请选择图片文件', 'error');
      return;
    }

    // 创建预览URL
    const url = URL.createObjectURL(file);
    setUploadedImageUrl(url);
    setUploadedFile(file); // 保存文件引用
    setInferenceResult(null); // 重置之前的结果
    setInferenceStatus('idle');
  };

  // 执行推理
  const handleInference = async (file: File) => {
    if (!selectedWeightId) {
      showNotification('请先选择权重', 'error');
      return;
    }

    setInferenceStatus('processing');
    setErrorMessage(null);

    try {
      const result = await inferenceService.predictWithImage(
        selectedWeightId,
        file,
        {
          confidence_threshold: confidenceThreshold,
          device: 'auto'
        }
      );
      setInferenceResult(result);
      setInferenceStatus('completed');
      showNotification('推理完成', 'success');
    } catch (error: any) {
      setInferenceStatus('error');
      setErrorMessage(error.message || '推理失败');
      showNotification(`推理失败: ${error.message || '未知错误'}`, 'error');
    }
  };

  // 处理上传按钮点击
  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  // 处理批量图片选择
  const handleBatchImageSelect = () => {
    batchInputRef.current?.click();
  };

  // 处理批量文件选择
  const handleBatchFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    // 过滤只保留图片文件
    const imageFiles = Array.from(files).filter(file =>
      file.type.startsWith('image/')
    );

    if (imageFiles.length === 0) {
      showNotification('请选择图片文件', 'error');
      return;
    }

    // 创建预览URL
    const images = imageFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      name: file.name
    }));

    setBatchImages(images);
    setBatchResults(new Map()); // 清空之前的结果
    setBatchInferenceStatus('idle');
    setSelectedBatchImage(null);
    showNotification(`已加载 ${images.length} 张图片`, 'success');
  };

  // 批量推理处理
  const handleBatchInference = async () => {
    if (!selectedWeightId) {
      showNotification('请先选择权重', 'error');
      return;
    }

    if (batchImages.length === 0) {
      showNotification('请先上传图片', 'error');
      return;
    }

    setBatchInferenceStatus('processing');
    const results = new Map<number, InferencePredictResponse>();

    try {
      for (let i = 0; i < batchImages.length; i++) {
        try {
          const result = await inferenceService.predictWithImage(
            selectedWeightId,
            batchImages[i].file,
            {
              confidence_threshold: confidenceThreshold,
              device: 'auto'
            }
          );
          results.set(i, result);
        } catch (error) {
          console.error(`图片 ${batchImages[i].name} 推理失败:`, error);
        }
      }

      setBatchResults(results);
      setBatchInferenceStatus('completed');
      showNotification(`批量推理完成，成功 ${results.size}/${batchImages.length}`, 'success');
    } catch (error: any) {
      setBatchInferenceStatus('idle');
      showNotification(`批量推理失败: ${error.message || '未知错误'}`, 'error');
    }
  };

  // 过滤结果（根据置信度阈值）
  const filteredResults = useMemo(() => {
    if (!inferenceResult || !inferenceResult.results) return [];
    return inferenceResult.results.filter(r =>
      (r as InferenceResult).confidence >= confidenceThreshold
    );
  }, [inferenceResult, confidenceThreshold]);

  // 计算检测框的样式（根据图像显示尺寸和原始尺寸计算正确的缩放比例）
  const getBboxStyle = useCallback((bbox: [number, number, number, number]) => {
    if (!imageDimensions || !imageRef.current) {
      return { display: 'none' };
    }

    const displayWidth = imageRef.current.clientWidth;
    const displayHeight = imageRef.current.clientHeight;

    const scaleX = displayWidth / imageDimensions.naturalWidth;
    const scaleY = displayHeight / imageDimensions.naturalHeight;

    return {
      left: `${bbox[0] * scaleX}px`,
      top: `${bbox[1] * scaleY}px`,
      width: `${(bbox[2] - bbox[0]) * scaleX}px`,
      height: `${(bbox[3] - bbox[1]) * scaleY}px`,
    };
  }, [imageDimensions]);

  // 下载检测结果 JSON
  const handleDownloadJson = useCallback(() => {
    if (!inferenceResult) return;

    const data = {
      task_type: inferenceResult.task_type,
      results: inferenceResult.results,
      metrics: inferenceResult.metrics,
      image_path: inferenceResult.image_path,
      weight_id: inferenceResult.weight_id,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inference_result_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('检测结果已下载', 'success');
  }, [inferenceResult]);

  // 下载标注图片（使用 Canvas 绘制检测框）
  const handleDownloadAnnotatedImage = useCallback(() => {
    if (!inferenceResult || !uploadedImageUrl || !imageRef.current) return;

    const img = imageRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 绘制原始图片
    ctx.drawImage(img, 0, 0);

    // 绘制检测框
    if (actualTaskType === 'detection' && filteredResults.length > 0) {
      filteredResults.forEach((result: any) => {
        const bbox = result.bbox as [number, number, number, number];

        // 绘制矩形框
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 3;
        ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);

        // 绘制标签背景
        const label = `${result.label} ${(result.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 16px Arial';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = '#00ffff';
        ctx.fillRect(bbox[0], bbox[1] - 24, textWidth + 8, 24);

        // 绘制标签文字
        ctx.fillStyle = '#000000';
        ctx.fillText(label, bbox[0] + 4, bbox[1] - 6);
      });
    }

    // 下载图片
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `annotated_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      showNotification('标注图片已下载', 'success');
    }, 'image/png');
  }, [inferenceResult, uploadedImageUrl, imageRef, actualTaskType, filteredResults]);

  const handleScanCameras = async () => {
      setStreamStatus('scanning');

      try {
          // 请求摄像头权限并获取设备列表
          // 先请求一次权限以确保能够获取设备标签
          await navigator.mediaDevices.getUserMedia({ video: true });

          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices
              .filter(device => device.kind === 'videoinput')
              .map(device => ({
                  id: device.deviceId,
                  label: device.label || `摄像头 ${device.deviceId.slice(0, 8)}`
              }));

          if (videoDevices.length > 0) {
              setCameras(videoDevices);
              setSelectedCamera(videoDevices[0].id);
              setStreamStatus('ready');
              showNotification(`已识别 ${videoDevices.length} 个视频输入设备`, "success");
          } else {
              setStreamStatus('idle');
              showNotification("未检测到摄像头设备", "error");
          }
      } catch (error: any) {
          console.error('摄像头检测失败:', error);
          setStreamStatus('idle');
          if (error.name === 'NotAllowedError') {
              showNotification("请允许访问摄像头权限", "error");
          } else if (error.name === 'NotFoundError') {
              showNotification("未检测到摄像头设备", "error");
          } else {
              showNotification("摄像头检测失败: " + error.message, "error");
          }
      }
  };

  const handleToggleStream = async () => {
      if (streamStatus === 'idle') {
          showNotification("请先连接摄像头设备", "error");
          return;
      }

      if (streamStatus === 'ready') {
          // 启动视频流
          try {
              const stream = await navigator.mediaDevices.getUserMedia({
                  video: {
                      deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
                      width: { ideal: 1280 },
                      height: { ideal: 720 }
                  }
              });

              setVideoStreamRef(stream);
              setStreamStatus('live');

              // 将视频流设置到 video 元素
              if (videoRef.current) {
                  videoRef.current.srcObject = stream;
              }

              showNotification("视频流推理已启动", "success");
          } catch (error: any) {
              console.error('启动视频流失败:', error);
              showNotification("启动视频流失败: " + error.message, "error");
          }
      } else if (streamStatus === 'live') {
          // 停止视频流
          if (videoStreamRef) {
              videoStreamRef.getTracks().forEach(track => track.stop());
              setVideoStreamRef(null);
          }
          if (videoRef.current) {
              videoRef.current.srcObject = null;
          }
          setStreamStatus('ready');
          showNotification("视频流已暂停", "info");
      }
  };

  // --- Dynamic Result Renderer ---
  const renderResults = () => {
    // 处理中状态
    if (inferenceStatus === 'processing') {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Loader2 size={32} className="mx-auto mb-3 animate-spin text-cyan-500" />
            <p className="text-sm text-slate-400">推理中...</p>
          </div>
        </div>
      );
    }

    // 错误状态
    if (inferenceStatus === 'error') {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <AlertTriangle size={32} className="mx-auto mb-3 text-rose-500" />
            <p className="text-sm text-rose-400">{errorMessage || '推理失败'}</p>
          </div>
        </div>
      );
    }

    // 空状态
    if (inferenceStatus === 'idle' || !inferenceResult) {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Image size={32} className="mx-auto mb-3 text-slate-600" />
            <p className="text-sm text-slate-500">请上传图片进行推理</p>
          </div>
        </div>
      );
    }

    // 显示真实结果
    const results = filteredResults;

    if (actualTaskType === 'detection') {
      return (
        <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
          {results.length === 0 ? (
            <p className="text-sm text-slate-500 text-center">未检测到目标</p>
          ) : (
            results.map((result: any, idx: number) => (
              <div key={idx} className="p-3 bg-slate-900/50 border border-slate-800 rounded-lg flex justify-between items-center hover:bg-slate-800 transition-colors cursor-pointer group">
                <div className="flex items-center">
                  <div className="w-2 h-2 rounded-full bg-cyan-500 mr-3"></div>
                  <span className="text-sm text-slate-300 font-medium">{result.label || `class_${result.class_id}`}</span>
                </div>
                <span className="font-mono text-xs text-slate-500 group-hover:text-white transition-colors">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))
          )}
        </div>
      );
    } else if (actualTaskType === 'classification') {
      return (
        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
          {results.length === 0 ? (
            <p className="text-sm text-slate-500 text-center">无分类结果</p>
          ) : (
            results.map((result: any, idx: number) => (
              <div key={idx}>
                <div className="flex justify-between text-xs mb-1">
                  <span className={`font-medium ${idx === 0 ? 'text-white' : 'text-slate-400'}`}>
                    {result.label || `class_${result.class_id}`}
                  </span>
                  <span className="font-mono text-slate-500">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${idx === 0 ? 'bg-cyan-500' : 'bg-slate-600'}`} style={{ width: `${result.confidence * 100}%` }}></div>
                </div>
              </div>
            ))
          )}
        </div>
      );
    }
  };

  return (
    <div className="h-full flex flex-col p-6 gap-6 relative">
        {/* Local Notification */}
        {notification && (
            <div className={`absolute top-6 left-1/2 -translate-x-1/2 z-[50] px-4 py-2 rounded-lg shadow-lg border flex items-center animate-in fade-in slide-in-from-top-2 duration-200 ${
                notification.type === 'error' ? 'bg-rose-900/90 border-rose-500 text-white' : 
                notification.type === 'success' ? 'bg-emerald-900/90 border-emerald-500 text-white' :
                'bg-cyan-900/90 border-cyan-500 text-white'
            }`}>
            {notification.type === 'error' ? <AlertTriangle size={16} className="mr-2" /> : 
             notification.type === 'success' ? <CheckCircle size={16} className="mr-2" /> :
             <Info size={16} className="mr-2" />
            }
            <span className="text-sm font-medium">{notification.msg}</span>
            </div>
        )}

       {/* Top Bar: Controls & Model Select */}
       <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col md:flex-row justify-between items-center gap-4 z-40">
          <div className="flex items-center space-x-4">
            <h2 className="text-xl font-bold text-white mr-4">推理实验室</h2>
            
            {/* Mode Switcher */}
            <div className={`flex bg-slate-900 rounded-lg p-1 border border-slate-800 ${isLocked ? 'opacity-50 pointer-events-none' : ''}`}>
               <button 
                  onClick={() => { setMode('single'); setSelectedBatchImage(null); }}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-colors ${mode === 'single' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
               >
                  <Image size={16} className="mr-2" /> 单图
               </button>
               <button
                  onClick={() => { setMode('batch'); setSelectedBatchImage(null); }}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-colors ${mode === 'batch' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
               >
                  <Layers size={16} className="mr-2" /> 批量
               </button>
               <button 
                  onClick={() => { setMode('stream'); setSelectedBatchImage(null); }}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-colors ${mode === 'stream' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
               >
                  <Video size={16} className="mr-2" /> 视频流
               </button>
            </div>

            {/* Batch Back Button - Repositioned Here */}
            {mode === 'batch' && selectedBatchImage !== null && (
                <div className="h-8 w-px bg-slate-800 mx-2"></div>
            )}
            {mode === 'batch' && selectedBatchImage !== null && (
               <button onClick={() => setSelectedBatchImage(null)} className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded-lg border border-slate-700 flex items-center transition-all">
                  <ArrowLeft size={14} className="mr-2" /> 返回概览
               </button>
            )}
          </div>

          <div className="flex items-center space-x-3 w-full md:w-auto">
             {/* 隐藏的文件输入 - 单图 */}
             <input
               ref={fileInputRef}
               type="file"
               accept="image/*"
               onChange={handleImageSelect}
               className="hidden"
             />

             {/* 隐藏的多图输入 - 批量 */}
             <input
               ref={batchInputRef}
               type="file"
               accept="image/*"
               multiple
               onChange={handleBatchFileChange}
               className="hidden"
             />

             <div className={`relative pt-2 ${isLocked ? 'opacity-50 pointer-events-none' : ''}`}>
                <label className="absolute -top-1 left-2 bg-slate-900 px-1 text-[10px] text-slate-500">加载权重</label>
                <WeightTreeSelect
                  options={weightTree}
                  value={selectedWeightId}
                  onChange={setSelectedWeightId}
                  disabled={isLocked || weightsLoading}
                  placeholder={weightsLoading ? '加载中...' : '请选择权重'}
                  className="flex-1 md:w-64"
                />
                {isLocked && <Lock size={12} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500" />}
             </div>

             {/* Action Button based on mode */}
             {mode === 'single' && (
               <>
                 <button
                   onClick={handleUploadClick}
                   disabled={inferenceStatus === 'processing'}
                   className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                 >
                    <Upload size={16} className="mr-2" /> 上传图片
                 </button>
                 {/* 开始推理按钮 - 只在有上传图片时显示 */}
                 {uploadedImageUrl && inferenceStatus !== 'processing' && (
                   <button
                     onClick={() => uploadedFile && handleInference(uploadedFile)}
                     disabled={!uploadedFile || !selectedWeightId}
                     className="flex items-center px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded-lg transition-colors disabled:cursor-not-allowed"
                   >
                      <Zap size={16} className="mr-2" /> 开始推理
                   </button>
                 )}
                 {/* 推理中状态 */}
                 {inferenceStatus === 'processing' && (
                   <div className="flex items-center px-4 py-2 bg-slate-800 text-slate-300 text-sm font-medium rounded-lg border border-slate-700">
                      <Loader2 size={16} className="mr-2 animate-spin text-cyan-500" /> 推理中
                   </div>
                 )}
               </>
             )}
             {mode === 'batch' && !selectedBatchImage && (
               <>
                 <button
                   onClick={handleBatchImageSelect}
                   disabled={batchInferenceStatus === 'processing'}
                   className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                 >
                    <Upload size={16} className="mr-2" /> 选择图片
                 </button>
                 {/* 批量开始推理按钮 */}
                 {batchImages.length > 0 && batchInferenceStatus !== 'processing' && (
                   <button
                     onClick={handleBatchInference}
                     disabled={!selectedWeightId || batchImages.length === 0}
                     className="flex items-center px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded-lg transition-colors disabled:cursor-not-allowed"
                   >
                      <Zap size={16} className="mr-2" /> 开始推理
                   </button>
                 )}
                 {/* 批量推理中状态 */}
                 {batchInferenceStatus === 'processing' && (
                   <div className="flex items-center px-4 py-2 bg-slate-800 text-slate-300 text-sm font-medium rounded-lg border border-slate-700">
                      <Loader2 size={16} className="mr-2 animate-spin text-cyan-500" /> 推理中 ({batchResults.size}/{batchImages.length})
                   </div>
                 )}
                 {/* 批量推理完成状态 */}
                 {batchInferenceStatus === 'completed' && (
                   <span className="text-sm text-emerald-400">完成 {batchResults.size}/{batchImages.length}</span>
                 )}
               </>
             )}
             
             {/* Video Stream Controls */}
             {mode === 'stream' && (
               <div className="flex items-center space-x-2">
                   {/* Connection Button / Dropdown */}
                   {streamStatus === 'idle' || streamStatus === 'scanning' ? (
                       <button
                            onClick={handleScanCameras}
                            disabled={streamStatus === 'scanning'}
                            className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-lg border border-slate-600 transition-colors"
                       >
                            {streamStatus === 'scanning' ? (
                                <><RefreshCw size={16} className="mr-2 animate-spin" /> 正在识别...</>
                            ) : (
                                <><Camera size={16} className="mr-2" /> 连接摄像头</>
                            )}
                       </button>
                   ) : (
                       <div className={`relative ${isLocked ? 'opacity-50 pointer-events-none' : ''}`}>
                           <select
                                value={selectedCamera}
                                onChange={(e) => setSelectedCamera(e.target.value)}
                                disabled={isLocked}
                                className="appearance-none bg-slate-900 border border-emerald-500/50 text-emerald-400 text-sm rounded-lg pl-3 pr-8 py-2 outline-none focus:border-emerald-400 cursor-pointer disabled:cursor-not-allowed"
                           >
                               {cameras.map(cam => (
                                   <option key={cam.id} value={cam.id}>{cam.label}</option>
                               ))}
                           </select>
                           <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-emerald-500 pointer-events-none" />
                       </div>
                   )}

                   {/* Mirror Toggle Button */}
                   {streamStatus === 'ready' || streamStatus === 'live' ? (
                       <button
                            onClick={() => setIsMirrored(!isMirrored)}
                            className={`flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all ${
                                isMirrored
                                ? 'bg-violet-600 hover:bg-violet-500 text-white'
                                : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                            }`}
                            title={isMirrored ? '已镜像' : '未镜像'}
                       >
                            <FlipHorizontal size={16} />
                       </button>
                   ) : null}

                   {/* Start/Stop Controls (Only when ready/live) */}
                   {(streamStatus === 'ready' || streamStatus === 'live') && (
                       <button
                            onClick={handleToggleStream}
                            className={`flex items-center px-4 py-2 text-white text-sm font-bold rounded-lg shadow-lg transition-all ${
                                streamStatus === 'live'
                                ? 'bg-amber-600 hover:bg-amber-500 shadow-amber-900/20'
                                : 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-900/20'
                            }`}
                       >
                            {streamStatus === 'live' ? (
                                <><Pause size={16} className="mr-2 fill-current" /> 暂停推理</>
                            ) : (
                                <><Play size={16} className="mr-2 fill-current" /> 开始推理</>
                            )}
                       </button>
                   )}
               </div>
             )}
          </div>
       </div>

       <div className="flex-1 flex gap-6 min-h-0">
          {/* Main Viewport */}
          <div className="flex-1 flex flex-col gap-4 min-h-0">
             <div className="flex-1 glass-panel rounded-xl border border-slate-800/60 p-4 flex items-center justify-center relative overflow-hidden bg-slate-950">
                
                {/* Content Area */}
                {mode === 'batch' && selectedBatchImage === null ? (
                    batchImages.length === 0 ? (
                        <div className="flex flex-col items-center text-slate-600">
                            <Layers size={64} className="mb-4 opacity-20" />
                            <p className="text-sm">点击上方按钮选择多张图片</p>
                        </div>
                    ) : (
                        <div className="w-full h-full overflow-y-auto grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4 content-start custom-scrollbar p-2">
                            {batchImages.map((img, idx) => {
                              const hasResult = batchResults.has(idx);
                              const isProcessing = batchInferenceStatus === 'processing' && !hasResult;
                              return (
                                <div
                                    key={idx}
                                    onClick={() => {
                                      setSelectedBatchImage(idx);
                                      // 设置当前推理结果用于右侧面板显示
                                      const result = batchResults.get(idx);
                                      setInferenceResult(result || null);
                                      setInferenceStatus(result ? 'completed' : 'idle');
                                    }}
                                    className={`aspect-square bg-slate-900 rounded border relative group cursor-pointer hover:scale-105 transition-all overflow-hidden ${
                                      hasResult ? 'border-emerald-500' : 'border-slate-800 hover:border-cyan-500'
                                    }`}
                                >
                                    <img src={img.preview} className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity" />
                                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-[10px] p-1 truncate opacity-0 group-hover:opacity-100 transition-opacity">
                                        {img.name}
                                    </div>
                                    {/* 推理状态标记 */}
                                    {hasResult && (
                                      <div className="absolute top-1 right-1 w-5 h-5 bg-emerald-500 rounded-full flex items-center justify-center">
                                        <CheckCircle size={12} className="text-white" />
                                      </div>
                                    )}
                                    {isProcessing && (
                                      <div className="absolute top-1 right-1 w-5 h-5 bg-cyan-500 rounded-full flex items-center justify-center">
                                        <Loader2 size={12} className="text-white animate-spin" />
                                      </div>
                                    )}
                                </div>
                              );
                            })}
                        </div>
                    )
                ) : selectedBatchImage !== null ? (
                    // 批量单图查看
                    <div className="relative w-full h-full flex items-center justify-center">
                        <img
                            src={batchImages[selectedBatchImage]?.preview}
                            className="max-w-full max-h-full object-contain"
                            alt={batchImages[selectedBatchImage]?.name}
                        />
                    </div>
                ) : (
                    <div className="relative w-full h-full flex items-center justify-center">
                        {/* STREAM MODE PLACEHOLDER / FEED */}
                        {mode === 'stream' ? (
                            streamStatus === 'idle' || streamStatus === 'scanning' ? (
                                <div className="flex flex-col items-center text-slate-600">
                                    <Camera size={64} className="mb-4 opacity-20" />
                                    <p className="text-sm">
                                        {streamStatus === 'scanning' ? '正在检测摄像头...' : '未连接视频输入源'}
                                    </p>
                                </div>
                            ) : (
                                <div className="relative w-full h-full flex items-center justify-center bg-black rounded-lg overflow-hidden border border-slate-800">
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className={`w-full h-full object-contain ${isMirrored ? 'scale-x-[-1]' : ''}`}
                                    />
                                    {streamStatus === 'live' && (
                                        <div className="absolute top-4 right-4 flex items-center space-x-2 bg-black/60 backdrop-blur px-3 py-1.5 rounded-full border border-red-500/30">
                                            <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse shadow-[0_0_8px_#ef4444]"></div>
                                            <span className="text-xs font-bold text-red-400">LIVE</span>
                                        </div>
                                    )}
                                    {/* Status LED Indicator */}
                                    <div className="absolute top-4 left-4 flex items-center space-x-2">
                                        <div className={`w-3 h-3 rounded-full shadow-lg border border-white/10 ${
                                            streamStatus === 'live' ? 'bg-emerald-500 shadow-[0_0_10px_#10b981]' :
                                            streamStatus === 'ready' ? 'bg-amber-500 shadow-[0_0_5px_#f59e0b]' : 'bg-slate-600'
                                        }`}></div>
                                        <span className="text-xs font-mono text-slate-300 bg-black/40 px-2 py-0.5 rounded">
                                            {streamStatus === 'live' ? 'Streaming' : 'Standby'}
                                        </span>
                                    </div>
                                </div>
                            )
                        ) : (
                            // IMAGE / BATCH SINGLE VIEW
                            <div className="relative w-full h-full flex items-center justify-center">
                                {uploadedImageUrl ? (
                                  <img
                                    ref={imageRef}
                                    src={uploadedImageUrl}
                                    className="max-w-full max-h-full object-contain"
                                    alt="Inference Input"
                                    onLoad={(e) => {
                                      const img = e.target as HTMLImageElement;
                                      setImageDimensions({ naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight });
                                    }}
                                  />
                                ) : (
                                  <div className="text-center text-slate-600">
                                    <Image size={64} className="mx-auto mb-4 opacity-20" />
                                    <p className="text-sm">请上传图片进行推理</p>
                                  </div>
                                )}
                                {/* 显示检测框（如果有结果） */}
                                {inferenceResult && actualTaskType === 'detection' && filteredResults.length > 0 && (
                                  <>
                                    {filteredResults.map((result: any, idx: number) => (
                                      <div
                                        key={idx}
                                        className="absolute border-2 border-cyan-400 bg-cyan-400/10 hover:bg-cyan-400/20 transition-colors cursor-pointer"
                                        style={getBboxStyle(result.bbox)}
                                      >
                                        <span className="absolute -top-5 left-0 bg-cyan-500 text-black text-[10px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap">
                                          {result.label} {(result.confidence * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    ))}
                                  </>
                                )}
                            </div>
                        )}
                    </div>
                )}
             </div>
             
             {/* Info Strip: Metrics (Single/Batch) OR Stream Stats */}
             {(mode !== 'batch' || selectedBatchImage !== null) && (
                <div className="h-20 glass-panel rounded-xl border border-slate-800 flex items-center justify-around px-6 shrink-0">
                    <div className="text-center group">
                        <div className="text-xs text-slate-500 uppercase mb-1 flex items-center justify-center group-hover:text-purple-400 transition-colors">
                            <Cpu size={12} className="mr-1.5" /> 推理设备
                        </div>
                        <div className="text-lg font-mono text-purple-400 font-bold">
                          {inferenceResult?.metrics?.device || 'Auto'}
                        </div>
                    </div>
                    <div className="w-px h-10 bg-slate-800"></div>
                    <div className="text-center group">
                        <div className="text-xs text-slate-500 uppercase mb-1 flex items-center justify-center group-hover:text-cyan-400 transition-colors">
                            <Clock size={12} className="mr-1.5" /> 推理耗时
                        </div>
                        <div className="text-xl font-mono text-cyan-400">
                          {inferenceResult?.metrics?.inference_time ? (
                            <>{inferenceResult.metrics.inference_time.toFixed(1)}<span className="text-sm text-slate-600 ml-1">ms</span></>
                          ) : (
                            <span className="text-slate-600">-</span>
                          )}
                        </div>
                    </div>
                    {/* Additional stat for Stream Mode */}
                    {mode === 'stream' && (
                        <>
                            <div className="w-px h-10 bg-slate-800"></div>
                            <div className="text-center group">
                                <div className="text-xs text-slate-500 uppercase mb-1 flex items-center justify-center group-hover:text-emerald-400 transition-colors">
                                    <Activity size={12} className="mr-1.5" /> 帧率 (FPS)
                                </div>
                                <div className="text-xl font-mono text-emerald-400">
                                    {streamStatus === 'live' ? '62' : '-'}
                                </div>
                            </div>
                        </>
                    )}
                </div>
             )}
          </div>

          {/* Results Sidebar (Standardized) */}
          {(mode !== 'batch' || selectedBatchImage !== null) && (
            <div className="w-80 bg-slate-950 border-l border-slate-800 rounded-xl flex flex-col overflow-hidden shrink-0">
               <div className="p-4 border-b border-slate-800 bg-slate-900/50">
                 <h3 className="font-bold text-white flex items-center capitalize">
                   {actualTaskType === 'detection' && <FileText size={18} className="mr-2 text-slate-400" />}
                   {actualTaskType === 'classification' && <Activity size={18} className="mr-2 text-slate-400" />}
                   {actualTaskType} Results
                 </h3>
               </div>
               
               {/* Dynamic Result List */}
               {renderResults()}

               {/* Unified Download Area (Conditionally Hidden for Stream) */}
               {mode !== 'stream' && (
                   <div className="p-4 border-t border-slate-800 space-y-2">
                     <button
                       onClick={handleDownloadJson}
                       disabled={!inferenceResult}
                       className="w-full py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:text-slate-600 text-slate-300 disabled:cursor-not-allowed text-xs font-bold rounded border border-slate-700 transition-colors flex items-center justify-center"
                     >
                       <Download size={14} className="mr-2" /> 下载检测结果 (JSON)
                     </button>
                     <button
                       onClick={handleDownloadAnnotatedImage}
                       disabled={!inferenceResult || !uploadedImageUrl}
                       className="w-full py-2 bg-cyan-900/30 hover:bg-cyan-900/50 disabled:bg-slate-900 disabled:text-slate-600 text-cyan-400 disabled:cursor-not-allowed text-xs font-bold rounded border border-cyan-800 transition-colors flex items-center justify-center"
                     >
                       <Image size={14} className="mr-2" /> 下载标注图片
                     </button>
                   </div>
               )}
            </div>
          )}
       </div>
    </div>
  );
};

export default InferenceView;