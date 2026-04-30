from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COLAB_ROOT = "/content"
COLAB_DRIVE_ROOT = "/content/drive"


@dataclass
class LoaderStorage:
    """Access notebook data from either Google Colab Drive or S3.

    The backend is selected entirely from `root`:

    - `s3://...` uses S3 via `s3fs`
    - `/content/...` uses the Colab filesystem and mounts Drive when needed

    Additional path fragments are appended onto `root` by `path()` and the
    read/write helpers.

    Attributes:
        root: Base filesystem root or S3 URI.
        mount_google_drive: Whether Google Drive should be mounted
            automatically when `root` points into `/content/drive`.
        fs: An `s3fs.S3FileSystem` instance when S3 is selected, otherwise
            `None`.
        storage_options: S3 credentials forwarded to pandas I/O methods when
            S3 is selected.
    """

    root: str
    mount_google_drive: bool = True
    fs: Any = field(init=False, default=None)
    storage_options: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the root and initialize the selected backend."""
        self.root = self._normalize_root(self.root)
        self.storage_options = self._build_storage_options()
        self.fs = self._build_filesystem()
        self._maybe_mount_google_drive()

    def path(self, *parts: str) -> str:
        """Join path fragments onto the configured root.

        Args:
            *parts: Relative path fragments to append to `root`.

        Returns:
            The combined local path or S3 URI.
        """
        return self._join(self.root, *parts)

    def read_csv(self, *parts: str, **kwargs: Any):
        """Read a CSV file relative to the configured root.

        Args:
            *parts: Relative path fragments appended to `root`.
            **kwargs: Extra keyword arguments forwarded to `pandas.read_csv`.

        Returns:
            The loaded pandas `DataFrame`.
        """
        import pandas as pd

        path = self.path(*parts)
        return pd.read_csv(path, **self._with_storage_options(path, kwargs))

    def write_parquet(self, dataframe: Any, *parts: str, **kwargs: Any) -> None:
        """Write a parquet file relative to the configured root.

        Args:
            dataframe: DataFrame-like object to persist.
            *parts: Relative path fragments appended to `root`.
            **kwargs: Extra keyword arguments forwarded to
                `DataFrame.to_parquet`.
        """
        path = self.path(*parts)
        dataframe.to_parquet(path, **self._with_storage_options(path, kwargs))

    def _build_storage_options(self) -> dict[str, Any]:
        """Build pandas storage options when the root targets S3.

        Returns:
            A dictionary suitable for pandas `storage_options`, or an empty
            dictionary for Colab/local paths.

        Raises:
            KeyError: If the root targets S3 but the required credentials are
                missing from the environment.
        """
        if not self._is_s3_root():
            return {}

        required = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "MLFLOW_S3_ENDPOINT_URL",
        ]
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise KeyError(
                f"Missing S3 environment variables: {', '.join(sorted(missing))}"
            )

        return {
            "key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
            "client_kwargs": {"endpoint_url": os.environ["MLFLOW_S3_ENDPOINT_URL"]},
        }

    def _build_filesystem(self):
        """Create an `s3fs` client when the root points at S3."""
        if not self._is_s3_root():
            return None

        import s3fs

        return s3fs.S3FileSystem(**self.storage_options)

    def _maybe_mount_google_drive(self) -> None:
        """Mount Google Drive when the root points into Colab Drive."""
        if not self.mount_google_drive or not self._uses_colab_drive():
            return

        try:
            from google.colab import drive
        except ImportError:
            return

        drive.mount(COLAB_DRIVE_ROOT)

    def _is_s3_root(self) -> bool:
        """Return whether the configured root selects the S3 backend."""
        return self.root.startswith("s3://")

    def _uses_colab_drive(self) -> bool:
        """Return whether the configured root points into Colab Drive."""
        return self.root == COLAB_DRIVE_ROOT or self.root.startswith(
            f"{COLAB_DRIVE_ROOT}/"
        )

    def _with_storage_options(
        self, path: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach S3 storage options to pandas I/O keyword arguments.

        Args:
            path: Fully resolved path targeted by the pandas operation.
            kwargs: Caller-provided pandas keyword arguments.

        Returns:
            The original keyword arguments for non-S3 paths or when
            `storage_options` were already provided. Otherwise, a copied
            mapping with this instance's S3 options attached.
        """
        if not path.startswith("s3://") or "storage_options" in kwargs:
            return kwargs

        merged_kwargs = dict(kwargs)
        merged_kwargs["storage_options"] = dict(self.storage_options)
        return merged_kwargs

    @staticmethod
    def _normalize_root(root: str) -> str:
        """Normalize the configured root without changing its backend.

        Args:
            root: Base local path or S3 URI.

        Returns:
            A normalized local path string or S3 URI without a trailing slash.

        Raises:
            ValueError: If `root` is empty.
        """
        if not root:
            raise ValueError("A storage root is required")

        if root.startswith("s3://"):
            return root.rstrip("/")

        return str(Path(root))

    @staticmethod
    def _join(root: str, *parts: str) -> str:
        """Append relative path fragments onto a local root or S3 URI.

        Args:
            root: Base local path or S3 URI.
            *parts: Relative path fragments to append.

        Returns:
            The combined path or URI.
        """
        clean_parts = [str(part).strip("/") for part in parts if str(part)]
        if root.startswith("s3://"):
            if not clean_parts:
                return root
            return f"{root.rstrip('/')}/{'/'.join(clean_parts)}"

        return str(Path(root).joinpath(*clean_parts))
