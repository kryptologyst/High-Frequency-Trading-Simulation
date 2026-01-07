"""Utility functions for device management and seeding."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device_preference: Preferred device ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: The selected device
    """
    if device_preference == "cpu":
        return torch.device("cpu")
    elif device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage for display.
    
    Args:
        value: Value to format as percentage
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"
