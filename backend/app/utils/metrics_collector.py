"""
系统指标收集器
收集系统资源使用情况（GPU、CPU、内存）并通过WebSocket推送
"""

import asyncio
import psutil
from loguru import logger
from datetime import datetime
from typing import Optional
import json

try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("pynvml未安装，GPU监控功能将不可用")


class MetricsCollector:
    """系统指标收集器"""

    def __init__(self):
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None
        self.collection_interval = 1.0  # 1秒收集一次（1Hz）

        # GPU初始化
        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU监控已初始化，检测到 {self.gpu_count} 个GPU")
            except Exception as e:
                logger.error(f"GPU初始化失败: {e}")
                self.gpu_available = False
                self.gpu_count = 0
        else:
            self.gpu_count = 0

    def get_cpu_metrics(self) -> dict:
        """
        获取CPU使用率

        Returns:
            CPU使用率（百分比）
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            return {
                "cpu_util": round(cpu_percent, 2),
                "cpu_count": cpu_count,
                "cpu_freq_current": round(cpu_freq.current, 2) if cpu_freq else None,
                "cpu_freq_max": round(cpu_freq.max, 2) if cpu_freq else None
            }
        except Exception as e:
            logger.error(f"获取CPU指标失败: {e}")
            return {"cpu_util": 0, "cpu_count": 0}

    def get_memory_metrics(self) -> dict:
        """
        获取内存使用情况

        Returns:
            内存使用信息（已用、总量、百分比）
        """
        try:
            mem = psutil.virtual_memory()

            return {
                "ram_used": round(mem.used / (1024**3), 2),  # GB
                "ram_total": round(mem.total / (1024**3), 2),  # GB
                "ram_percent": round(mem.percent, 2),
                "ram_available": round(mem.available / (1024**3), 2)  # GB
            }
        except Exception as e:
            logger.error(f"获取内存指标失败: {e}")
            return {"ram_used": 0, "ram_total": 0, "ram_percent": 0}

    def get_disk_metrics(self) -> dict:
        """
        获取磁盘使用情况

        Returns:
            磁盘使用信息
        """
        try:
            disk = psutil.disk_usage('/')

            return {
                "disk_used": round(disk.used / (1024**3), 2),  # GB
                "disk_total": round(disk.total / (1024**3), 2),  # GB
                "disk_percent": round(disk.percent, 2)
            }
        except Exception as e:
            logger.error(f"获取磁盘指标失败: {e}")
            return {"disk_used": 0, "disk_total": 0, "disk_percent": 0}

    def get_gpu_metrics(self) -> list:
        """
        获取GPU使用情况

        Returns:
            GPU使用信息列表（每个GPU一个字典）
        """
        if not self.gpu_available:
            return []

        gpu_metrics = []

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # 获取GPU使用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu

                # 获取温度
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )

                # 获取显存使用情况
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used = round(mem_info.used / (1024**3), 2)
                vram_total = round(mem_info.total / (1024**3), 2)
                vram_percent = round((mem_info.used / mem_info.total) * 100, 2)

                # 获取功耗
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # 转换为瓦特
                except:
                    power_usage = None

                # 获取GPU名称
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                gpu_metrics.append({
                    "gpu_id": i,
                    "gpu_name": name,
                    "gpu_util": gpu_util,
                    "gpu_temp": temperature,
                    "vram_used": vram_used,
                    "vram_total": vram_total,
                    "vram_percent": vram_percent,
                    "power_usage": round(power_usage, 2) if power_usage else None
                })

        except Exception as e:
            logger.error(f"获取GPU指标失败: {e}")

        return gpu_metrics

    def get_network_metrics(self) -> dict:
        """
        获取网络使用情况

        Returns:
            网络收发字节数
        """
        try:
            net_io = psutil.net_io_counters()

            return {
                "bytes_sent": round(net_io.bytes_sent / (1024**2), 2),  # MB
                "bytes_recv": round(net_io.bytes_recv / (1024**2), 2),  # MB
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            logger.error(f"获取网络指标失败: {e}")
            return {"bytes_sent": 0, "bytes_recv": 0}

    def collect_metrics(self) -> dict:
        """
        收集所有系统指标

        Returns:
            完整的系统指标字典
        """
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cpu": self.get_cpu_metrics(),
            "memory": self.get_memory_metrics(),
            "disk": self.get_disk_metrics(),
            "gpu": self.get_gpu_metrics(),
            "network": self.get_network_metrics()
        }

        return metrics

    async def start_collection(self, callback=None):
        """
        启动指标收集循环

        Args:
            callback: 回调函数，接收收集到的指标
        """
        if self.running:
            logger.warning("指标收集器已经在运行中")
            return

        self.running = True
        logger.info("启动系统指标收集器...")

        async def collection_loop():
            """指标收集循环"""
            while self.running:
                try:
                    # 收集指标
                    metrics = self.collect_metrics()

                    # 调用回调函数（如果有）
                    if callback:
                        await callback(metrics)

                    # 等待下一次收集
                    await asyncio.sleep(self.collection_interval)

                except Exception as e:
                    logger.error(f"指标收集错误: {e}")
                    await asyncio.sleep(self.collection_interval)

        # 创建后台任务
        self.collection_task = asyncio.create_task(collection_loop())

    async def stop_collection(self):
        """停止指标收集"""
        if not self.running:
            return

        logger.info("停止系统指标收集器...")
        self.running = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass


# 全局指标收集器实例
collector = MetricsCollector()
