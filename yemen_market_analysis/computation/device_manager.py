"""
Device management for computation acceleration in Yemen Market Analysis.
"""
import logging
import numpy as np
from typing import Any, Optional, Tuple, Union, List

from core.decorators import error_handler, performance_tracker
from core.exceptions import DeviceError

logger = logging.getLogger(__name__)

# Singleton instance
_device_manager = None


class DeviceManager:
    """Unified device abstraction for computation acceleration."""
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize device manager.
        
        Args:
            force_cpu: Whether to force CPU usage even if GPU is available
        """
        self._device = None
        self.device_type = "cpu"
        self.has_gpu = False
        self.is_mps = False
        self.force_cpu = force_cpu
        self._initialize_device()
    
    @error_handler
    def _initialize_device(self) -> None:
        """Initialize the best available computation device."""
        if self.force_cpu:
            logger.info("Forcing CPU usage as requested")
            self._device = None
            self.device_type = "cpu"
            return
            
        try:
            import torch
            
            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available() and self._test_device('cuda'):
                self.device_type = "cuda"
                self._device = torch.device("cuda")
                self.has_gpu = True
                logger.info("Using CUDA for GPU acceleration")
                return
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self._test_device('mps'):
                self.device_type = "mps"
                self._device = torch.device("mps")
                self.has_gpu = True
                self.is_mps = True
                logger.info("Using MPS for Apple Silicon GPU acceleration with automatic CPU fallback")
                return
            
            # Fall back to CPU
            self._device = torch.device("cpu")
            logger.info("Using CPU for computations (no GPU acceleration available)")
            
        except ImportError:
            logger.info("PyTorch not available. Using CPU implementations only.")
            self._device = None
    
    @error_handler(fallback_value=False)
    def _test_device(self, device_type: str) -> bool:
        """
        Test if a device is working properly.
        
        Args:
            device_type: Device type to test
            
        Returns:
            Whether the device is working
        """
        try:
            import torch
            
            device = torch.device(device_type)
            x = torch.randn(10, 10, device=device)
            y = torch.mm(x, x)
            return True
        except Exception as e:
            logger.warning(f"Device {device_type} test failed: {str(e)}")
            return False
    
    @property
    def device(self) -> Any:
        """Get the current device object."""
        return self._device
    
    @error_handler
    def to_device(self, tensor: Any) -> Any:
        """
        Move tensor to current device.
        
        Args:
            tensor: Tensor to move
            
        Returns:
            Tensor on device
        """
        import torch
        
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self._device)
        elif isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, device=self._device)
        else:
            return tensor
    
    @error_handler(fallback_value=None)
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert tensor to numpy array.
        
        Args:
            tensor: Tensor to convert
            
        Returns:
            Numpy array
        """
        import torch
        
        if isinstance(tensor, torch.Tensor):
            if self.is_mps and tensor.device.type == 'mps':
                return tensor.detach().cpu().numpy()
            return tensor.detach().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)
    
    @error_handler
    def matrix_multiply(self, a: Any, b: Any) -> np.ndarray:
        """
        Perform matrix multiplication with GPU acceleration if available.
        
        Args:
            a: First matrix
            b: Second matrix
            
        Returns:
            Result matrix as numpy array
        """
        if self.has_gpu:
            import torch
            
            try:
                a_t = self.to_device(a)
                b_t = self.to_device(b)
                result = torch.matmul(a_t, b_t)
                return self.to_numpy(result)
            except Exception as e:
                logger.warning(f"GPU matrix multiplication failed, falling back to CPU: {str(e)}")
        
        # CPU fallback
        return np.matmul(a, b)
    
    @error_handler
    def solve_linear_system(self, A: Any, b: Any) -> np.ndarray:
        """
        Solve linear system Ax = b with GPU acceleration if available.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector or matrix
            
        Returns:
            Solution vector or matrix as numpy array
        """
        if self.has_gpu:
            import torch
            
            try:
                A_t = self.to_device(A)
                b_t = self.to_device(b)
                
                # Handle case for vector b
                if b_t.dim() == 1:
                    b_t = b_t.unsqueeze(1)
                
                # Check if A is invertible
                if A_t.shape[0] == A_t.shape[1]:
                    result = torch.linalg.solve(A_t, b_t)
                else:
                    # Use least squares for non-square matrices
                    result, _ = torch.linalg.lstsq(A_t, b_t)
                
                if b.ndim == 1:
                    result = result.squeeze(1)
                
                return self.to_numpy(result)
            except Exception as e:
                logger.warning(f"GPU linear system solution failed, falling back to CPU: {str(e)}")
        
        # CPU fallback
        if A.shape[0] == A.shape[1]:
            return np.linalg.solve(A, b)
        else:
            return np.linalg.lstsq(A, b, rcond=None)[0]


def get_device_manager(force_cpu: bool = False) -> DeviceManager:
    """
    Get the global device manager instance.
    
    Args:
        force_cpu: Whether to force CPU usage
        
    Returns:
        DeviceManager instance
    """
    global _device_manager
    
    if _device_manager is None or _device_manager.force_cpu != force_cpu:
        _device_manager = DeviceManager(force_cpu=force_cpu)
        
    return _device_manager


def set_device_manager(force_cpu: bool = False) -> DeviceManager:
    """
    Set the global device manager instance.
    
    Args:
        force_cpu: Whether to force CPU usage
        
    Returns:
        DeviceManager instance
    """
    global _device_manager
    _device_manager = DeviceManager(force_cpu=force_cpu)
    return _device_manager