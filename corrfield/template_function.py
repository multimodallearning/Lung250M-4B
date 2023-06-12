import torch


def template_function(a: torch.Tensor, b: int,c: float = 1.5) -> torch.Tensor:
    r"""
    The doc string should contain a (brief) description of the function and
    a list of function arguments and the function return with their type
    and dimensions (for tensors).

    e.g. (below the function are two examples from torch.optim.Adam
    and torch.functional.interpolate):

    This is a dummy function that multiplies a torch tensor with an integer
    and a float number.
    It serves to define a function template.


    Parameters:
            a (Tensor): a 4D (B,C,H,W) tensor with an input image
            b (int): An integer parameter
            c (float): A float parameter (default: 1.5)

    Returns:
            multiplication (Tensor): a 4D (B,C,H,W) tensor


    Useful Links:

    docstring overview:

    https://www.geeksforgeeks.org/python-docstrings/

    python style guide and useful tips:

    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

    https://github.com/IgorSusmelj/pytorch-styleguide

    ToDos:
        optional ToDos if the function lacks certain functionality


    Example:
        This section can contain minimal working examples for function calls

        multiplication = template_function(torch.ones(1,dtype=torch.half),2,c=3)

            $ python template_function.py
    """
    # Error handling or warnings can be helpful for restricted parameter input
    assert (torch.is_tensor(a)), "First input not a tensor"
    if c < 0:
        raise ValueError('Example error for c<0')
    device = a.device  # make function device independent (i.e. put all tensors on input tensor  device)
    precision = a.dtype  # if possible also make function precision independent e.g. for torch amp
    d = torch.ones(1,dtype=precision).to(device)  # create tensors with correct precision on the correct device
    e = torch.ones(1).type(precision).to(device)  # also possible to use type
    multiplication = a*b*c*d*e
    return multiplication


# The following docstrings are copied (with slight alterations to amtch above example)
# from the definitions of torch.optim.Adam and torch.nn.functional.interpolate
# and serve as examples of how a docstring could also look like

# torch.optim.Adam docstring example
""" Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Parameters:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
       """

# torch.nn.functional.interpolate docstring example
""" Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

    Parameters:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation.  When `scale_factor` is passed as a parameter, it is used
            to compute the `output_size`.  If `recompute_scale_factor` is ```False`` or not specified,
            the passed-in `scale_factor` will be used in the interpolation computation.
            Otherwise, a new `scale_factor` will be computed based on the output and input sizes for
            use in the interpolation computation (i.e. the computation will be identical to if the computed
            `output_size` were passed-in explicitly).  Note that when `scale_factor` is floating-point,
            the recomputed scale_factor may differ from the one passed in due to rounding and precision
            issues.

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    .. warning::
        When scale_factor is specified, if recompute_scale_factor=True,
        scale_factor is used to compute the output_size which will then
        be used to infer new scales for the interpolation.
        The default behavior for recompute_scale_factor changed to False
        in 1.6.0, and scale_factor is used in the interpolation
        calculation.

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.
    """