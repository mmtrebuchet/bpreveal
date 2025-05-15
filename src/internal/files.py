"""Internal functions for file manipulation."""
import datetime
import h5py
import bpreveal
from bpreveal import logUtils


def addH5Metadata(h5fp: h5py.File, **kwargs: str) -> None:
    """Adds some metadata to your hdf5 file for reproducibility.

    :param h5fp: The opened hdf5 file that you want to add metadata to.
        This file must be opened for writing.
    :param kwargs: The attributes you want to store.

    This will create a group called ``metadata`` and it will store everything
    listed in ``kwargs``. It also stores the BPReveal version in ``bpreveal_version``,
    and the date of creation in ``created_date``.
    """
    logUtils.debug("Adding configuration data to hdf5 file.")
    grp = h5fp.create_group("metadata")
    grp.attrs["bpreveal_version"] = str(bpreveal.__version__)
    grp.attrs["created_date"] = str(datetime.datetime.today())
    for k, v in kwargs.items():
        grp.attrs[str(k)] = str(v)


# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
