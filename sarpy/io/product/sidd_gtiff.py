"""
Module for writing SIDD Geotiff files (following Sensor Independent
Derived Data (SIDD), Volume 3, GeoTIFF File Format Description
Document,Version 2.0, 2019-05-31)

Partly reused libtiff library tiff_image class for tiff interaction
but extended the tags list and added reading of metadata from SIDD

"""


__classification__ = "UNCLASSIFIED"
__author__ = "Leszek Lamentowski"

import datetime
import logging
import sys
import os
import time

from collections import namedtuple

import numpy

from libtiff.tiff_data import (
    tag_name2value,
    tag_value2type,
    tag_value2name,
    name2type,
    type2bytes,
    type2dtype,
)
from libtiff.tiff_image import TIFFentry, TIFFimage
from libtiff.utils import bytes2str

from .sidd import SIDDReader

SIDDDataCollection = namedtuple(
    "SIDDDataCollection", ["image_array", "sidd_gtiff_metadata"]
)


# Dictionary mapping displayed pixel type into
# (BitsPerSample,PhotometricInterpretation,SamplesPerPixel)
SIDDPixelType = dict(
    MONO8I=(8, 1, None),
    MONO8LU=(8, 1, None),
    MONO16I=(16, 1, 2),
    RGB8LU=(8, 3, None),
    RGB24I=([8, 8, 8], 2, 3),
)


def convert_sidd_nitf_gtiff(sidd_reader, target_filename, colormap=None):
    """
    Main function for converting SIDD from NITF format into GeoTIFF along
    the requirements of the SIDD standard Vol.3.
    Requires an instance of SIDDReader as the input, then GeoTIFF metadata
    is created and written to the given target_filename
    """

    assert isinstance(
        sidd_reader, SIDDReader
    ), "convert_sidd_nitf_gtiff requires SIDDReader object as an input"
    # Add trying to open that filename for writing and catch error if it fails

    sidd_metadata = sidd_reader.sidd_meta[0]

    sidd_gtiff_metadata = dict(
        ImageWidth=sidd_metadata.Measurement.PixelFootprint.Col,
        ImageLength=sidd_metadata.Measurement.PixelFootprint.Row,
        Compression=1,
        ImageDescription="SIDD GeoTIFF test image",
        Orientation=1,
        #        XResolution=sidd_metadata.Measurement.PlaneProjection.SampleSpacing.Col,
        #        YResolution=sidd_metadata.Measurement.PlaneProjection.SampleSpacing.Row,
        XResolution=sidd_metadata.ExploitationFeatures.Products[0].Resolution.Col,
        YResolution=sidd_metadata.ExploitationFeatures.Products[0].Resolution.Row,
        PlanarConfiguration=1,
        ResolutionUnit=1,
        Software=sidd_metadata.ProductCreation.ProcessorInformation.Application,
        DateTime="lol",  # convert datetime from SIDD metadata into YYYY:MM:DD HH:MM:SS + NULL
        Artist=sidd_metadata.ProductCreation.ProcessorInformation.Site,
    )

    # Date time:
    # conversion from 2021-06-30T08:07:10.986330Z  into YYYY:MM:DDHH:MM:SS\0
    seconds_since_epoch = (
        sidd_metadata.ProductCreation.ProcessorInformation.ProcessingDateTime
        - numpy.datetime64(0, "s")
    ) / numpy.timedelta64(1, "s")
    original_datetime = datetime.datetime.utcfromtimestamp(seconds_since_epoch)

    converted_datetime = original_datetime.strftime("%Y:%m:%d%H:%M:%S")
    date_time = dict(DateTime=converted_datetime)
    # Sample characterization:
    sample_format = dict(
        BitsPerSample=SIDDPixelType[sidd_metadata.Display.PixelType][0],
        PhotometricInterpretation=SIDDPixelType[sidd_metadata.Display.PixelType][1],
    )

    if SIDDPixelType[sidd_metadata.Display.PixelType][2] is not None:
        sample_format.update(
            {"SamplesPerPixel": SIDDPixelType[sidd_metadata.Display.PixelType][2]}
        )

    if sidd_metadata.Display.PixelType == "RGB8LU":
        assert isinstance(
            colormap, numpy.ndarray
        ), "For PixelType RGB8LU, a custom colormap has to be supplied as an numpy.ndarray"
        sample_format.update({"ColorMap": colormap.tobytes()})

    # GeoTIFF geo-referencing for Geodetic Grid Display
    # 1. the tie point is the position at LRLC
    # 2. the pixel scale is the resolution converted to arc-seconds
    # 3. KeyDirectoryTag is constructed along the spec of SIDD GeoTIFF
    # 4. GeoAsciiParamsTag is empty string
    model_tie_point = [
        0,
        0,
        0,
        sidd_metadata.GeoData.ImageCorners.get_array()[2].Lat,
        sidd_metadata.GeoData.ImageCorners.get_array()[2].Lon,
        0,
    ]

    mean_earth_radius = 6.3710e3
    pixel_scale_row = (
        sidd_metadata.Measurement.PlaneProjection.SampleSpacing.Row / mean_earth_radius
    )
    pixel_scale_col = (
        sidd_metadata.Measurement.PlaneProjection.SampleSpacing.Col / mean_earth_radius
    )
    pixel_scale = [numpy.rad2deg(pixel_scale_col), numpy.rad2deg(pixel_scale_row), 0]

    # See http://geotiff.maptools.org/spec/geotiff2.4.html for reference on this table
    key_directory = [1, 1, 2, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]

    geo_referencing = dict(
        ModelPixelScaleTag=pixel_scale,
        ModelTiepointTag=model_tie_point,
        GeoKeyDirectoryTag=key_directory,
        GeoAsciiParamsTag="",
    )

    # SICD and SIDD XML metadata
    des_headers = []

    for it in range(len(sidd_reader.nitf_details.des_segment_sizes)):
        des_headers.append(sidd_reader.nitf_details.get_des_bytes(it))
        logging.info("Added another set of ")

    logging.info("Extracted %d des XML segments", it)
    des_headers_joined = "\0".join(str(des_headers)) + "\0"
    geo_metadata = dict(Geo_Metadata=des_headers_joined.encode("ascii"))

    sidd_gtiff_metadata.update(date_time)
    sidd_gtiff_metadata.update(sample_format)
    sidd_gtiff_metadata.update(geo_referencing)
    sidd_gtiff_metadata.update(geo_metadata)
    # Add image_data to return values
    # Understand image_data shape relation to the length and width metadata elements

    gtiff_image = SIDDGTIFFimage(
        SIDDDataCollection(sidd_reader[:, :], sidd_gtiff_metadata)
    )
    gtiff_image.write_file(target_filename)


def extend_tags():
    """
    Setup custom tags collection for the SIDD GeoTIFF
    """

    # SIDD GeoTIFF 1.0 TIFF Tags
    _gtiff_tag_info = """
        ModelPixelScaleTag 830E LONG 12
        ModelTiepointTag 8482 LONG 12
        GeoKeyDirectoryTag 87AF SHORT 3
        GeoAsciiParamsTag 34737 ASCII
        Geo_Metadata C6DD ASCII"""

    _gtiff_tag_value2name = {}
    _gtiff_tag_name2value = {}
    _gtiff_tag_value2type = {}

    for line in _gtiff_tag_info.split("\n"):
        if not line or line.startswith("#"):
            continue
        clean_line = line.lstrip()
        n, h, t = clean_line.split()[:3]
        h = eval("0x" + h)
        _gtiff_tag_value2name[h] = n
        _gtiff_tag_value2type[h] = t
        _gtiff_tag_name2value[n] = h

    tag_value2name.update(_gtiff_tag_value2name)
    tag_name2value.update(_gtiff_tag_name2value)
    tag_value2type.update(_gtiff_tag_value2type)

    return tag_value2name, tag_name2value, tag_value2type


class SIDDGTIFFentry(TIFFentry):
    """Hold a IFD entry used by SIDDGTIFFimage."""

    def __init__(self, tag):
        _tag_value2name, _tag_name2value, _tag_value2type = extend_tags()

        if isinstance(tag, str):
            tag = _tag_name2value[tag]
        assert isinstance(tag, int), repr(tag)
        self.tag = tag
        self.type_name = _tag_value2type[tag]
        self.type = name2type[self.type_name]
        self.type_nbytes = type2bytes[self.type]
        self.type_dtype = type2dtype[self.type]
        self.tag_name = _tag_value2name.get(self.tag, "TAG%s" % (hex(self.tag),))

        self.record = numpy.zeros((12,), dtype=numpy.ubyte)
        self.record[:2].view(dtype=numpy.uint16)[0] = self.tag
        self.record[2:4].view(dtype=numpy.uint16)[0] = self.type
        self.values = []


class SIDDGTIFFimage(TIFFimage):
    """
    Hold an image stack that can be written to SIDD GeoTIFF file.
    """

    def __init__(self, data):
        """
        data : {list, SIDDDataCollection}
          Specify image data as a list of images or as an array with rank<=3.
            #TODO:
                Require valid SIDD nitf product as an input
                Write an external function to convert from SIDD metadata
                    to GEOTIFF metadata

        """
        # dtype = None
        if isinstance(data, list):
            for item in data:
                assert isinstance(item, SIDDDataCollection)
            self.data = data
        elif not isinstance(data, SIDDDataCollection):
            raise NotImplementedError(repr(type(data)))
        else:
            self.data = [data]

    # noinspection PyProtectedMember
    def write_file(
        self,
        filename,
        strip_size=2 ** 13,
        planar_config=1,
    ):
        """
        Write image data to TIFF file.
        Parameters
        ----------
        filename : str
        compression : {'none', 'lzw'}
        strip_size : int
          Specify the size of uncompressed strip.
        planar_config : int
        validate : bool
          When True then check compression by decompression.
        Returns
        -------
        compression : float
          Compression factor.
        """

        if os.path.splitext(filename)[1].lower() not in [".tif", ".tiff"]:
            filename += ".tif"

        logging.info("Writing TIFF records to %s\n", filename)

        # compute tif file size and create image file directories data
        image_directories = []
        total_size = 8
        data_size = 0
        image_data_size = 0

        for i, data_item in enumerate(self.data):
            image = data_item.image_array
            metadata = data_item.sidd_gtiff_metadata
            logging.info(
                "\r  creating records: %5s%% done  ", (int(100.0 * i / len(self.data)))
            )
            if image.dtype.kind == "V" and len(image.dtype.names) == 3:  # RGB image
                sample_format = dict(u=1, i=2, f=3, c=6).get(
                    image.dtype.fields[image.dtype.names[0]][0].kind
                )
                bits_per_sample = [
                    image.dtype.fields[f][0].itemsize * 8 for f in image.dtype.names
                ]
                samples_per_pixel = 3
                photometric_interpretation = 2
            else:  # gray scale image
                sample_format = dict(u=1, i=2, f=3, c=6).get(image.dtype.kind)
                bits_per_sample = image.dtype.itemsize * 8
                samples_per_pixel = 1
                photometric_interpretation = 1
            if sample_format is None:
                logging.warning(
                    "Warning(TIFFimage.write_file): unknown data kind %r, "
                    "mapping to void",
                    image.dtype.kind,
                )
                sample_format = 4

            length, width = image.shape
            bytes_per_row = width * image.dtype.itemsize
            rows_per_strip = length
            strips_per_image = 1

            assert bytes_per_row * rows_per_strip * strips_per_image >= image.nbytes

            entries = []
            for tagname, value in list(metadata.items()):
                entry = SIDDGTIFFentry(tagname)
                entry.add_value(value)
                entries.append(entry)
                total_size += 12 + entry.nbytes
                data_size += entry.nbytes

            strip_byte_counts = SIDDGTIFFentry("StripByteCounts")
            strip_offsets = SIDDGTIFFentry("StripOffsets")
            entries.append(strip_byte_counts)
            entries.append(strip_offsets)

            # strip_offsets and strip_byte_counts will be filled in the next
            # loop
            assert strip_byte_counts.type_nbytes <= 4
            assert strip_offsets.type_nbytes <= 4
            total_size += 2 * 12

            # image data:
            total_size += image.nbytes
            data_size += image.nbytes
            image_data_size += image.nbytes

            # records for nof IFD entries and offset to the next IFD:
            total_size += 2 + 4

            # entries must be sorted by tag number
            entries.sort(key=lambda x: x.tag)

            strip_info = (
                strip_offsets,
                strip_byte_counts,
                strips_per_image,
                rows_per_strip,
                bytes_per_row,
            )
            image_directories.append((entries, strip_info, image))

        tif = numpy.memmap(filename, dtype=numpy.ubyte, mode="w+", shape=(total_size,))
        logging.info("Opened new memmap at %s", filename)
        # noinspection PyProtectedMember
        def tif_write(_tif, _offset, _data):
            end = _offset + _data.nbytes
            if end > _tif.size:
                size_incr = int(float(end - _tif.size) / 1024 ** 2 + 1) * 1024 ** 2
                new_size = _tif.size + size_incr
                assert end <= new_size, repr((end, _tif.size, size_incr, new_size))
                # sys.stdout.write('resizing: %s -> %s\n' % (tif.size,
                # new_size))
                # tif.resize(end, refcheck=False)
                _base = _tif._mmap
                if _base is None:
                    _base = _tif.base
                _base.resize(new_size)
                new_tif = numpy.ndarray.__new__(
                    numpy.memmap, (_base.size(),), dtype=_tif.dtype, buffer=_base
                )
                new_tif._parent = _tif
                new_tif.__array_finalize__(_tif)
                _tif = new_tif
            _tif[_offset:end] = _data
            return _tif

        # write TIFF header
        tif[:2].view(dtype=numpy.uint16)[0] = 0x4949  # low-endian
        tif[2:4].view(dtype=numpy.uint16)[0] = 42  # magic number
        tif[4:8].view(dtype=numpy.uint32)[0] = 8  # offset to the first IFD

        offset = 8
        data_offset = total_size - data_size
        image_data_offset = total_size - image_data_size
        first_data_offset = data_offset
        first_image_data_offset = image_data_offset
        start_time = time.time()
        for i, (entries, strip_info, image) in enumerate(image_directories):
            (
                strip_offsets,
                strip_byte_counts,
                strips_per_image,
                rows_per_strip,
                bytes_per_row,
            ) = strip_info

            # write the nof IFD entries
            tif[offset : offset + 2].view(dtype=numpy.uint16)[0] = len(entries)
            offset += 2
            assert offset <= first_data_offset, repr((offset, first_data_offset))

            # write image data
            data = image.view(dtype=numpy.ubyte).reshape((image.nbytes,))

            for j in range(strips_per_image):
                c = rows_per_strip * bytes_per_row
                k = j * c
                c -= max((j + 1) * c - image.nbytes, 0)
                assert c > 0, repr(c)
                orig_strip = data[k : k + c]  # type: numpy.ndarray

                strip = orig_strip
                # print strip.size, strip.nbytes, strip.shape,
                # tif[image_data_offset:image_data_offset+strip.nbytes].shape
                strip_offsets.add_value(image_data_offset)
                strip_byte_counts.add_value(strip.nbytes)

                tif = tif_write(tif, image_data_offset, strip)
                image_data_offset += strip.nbytes
                # if j == 0:
                #     first = strip_offsets[0]
                # last = strip_offsets[-1] + strip_byte_counts[-1]

            # write IFD entries
            for entry in entries:
                data_size = entry.nbytes
                if data_size:
                    entry.set_offset(data_offset)
                    assert data_offset + data_size <= total_size, repr(
                        (data_offset + data_size, total_size)
                    )
                    r = entry.toarray(tif[data_offset : data_offset + data_size])
                    assert r.nbytes == data_size
                    data_offset += data_size
                    assert data_offset <= first_image_data_offset, repr(
                        (data_offset, first_image_data_offset, i)
                    )
                tif[offset : offset + 12] = entry.record
                offset += 12
                assert offset <= first_data_offset, repr((offset, first_data_offset, i))

            # write offset to the next IFD
            tif[offset : offset + 4].view(dtype=numpy.uint32)[0] = offset + 4
            offset += 4
            assert offset <= first_data_offset, repr((offset, first_data_offset))

            logging.info(
                "\r  filling records: %5s%% done (%s/s)%s"
                % (
                    int(100.0 * (i + 1) / len(image_directories)),
                    bytes2str(
                        int(
                            float(image_data_offset - first_image_data_offset)
                            / (time.time() - start_time)
                        )
                    ),
                    " " * 2,
                )
            )

        # last offset must be 0
        tif[offset - 4 : offset].view(dtype=numpy.uint32)[0] = 0

        del tif  # flushing
