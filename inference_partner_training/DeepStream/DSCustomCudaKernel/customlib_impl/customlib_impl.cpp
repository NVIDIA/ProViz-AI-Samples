/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"

#include "nvdscustomlib_base.hpp"
#include "simple_kernel.hpp"

#define FORMAT_NV12 "NV12"
#define FORMAT_RGBA "RGBA"

inline bool CHECK_(int e, int iLine, const char* szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile;
        exit(-1);
        return false;
    }
    return true;
}
#define ck(call) CHECK_(call, __LINE__, __FILE__)

/* This quark is required to identify NvDsMeta when iterating through
 * the buffer metadatas */
static GQuark _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);

/* Strcture used to share between the threads */
struct PacketInfo
{
    GstBuffer* inbuf;
    guint frame_num;
};

class SampleAlgorithm : public DSCustomLibraryBase
{
public:
    SampleAlgorithm()
    {
        m_vectorProperty.clear();
        outputthread_stopped = false;
    }

    /* Set Init Parameters */
    virtual bool SetInitParams(DSCustom_CreateParams* params);

    /* Set Custom Properties  of the library */
    virtual bool SetProperty(Property& prop);

    /* Pass GST events to the library */
    virtual bool HandleEvent(GstEvent* event);

    /* Process Incoming Buffer */
    virtual BufferResult ProcessBuffer(GstBuffer* inbuf);

    /* Retrun Compatible Caps */
    virtual GstCaps* GetCompatibleCaps(GstPadDirection direction, GstCaps* in_caps, GstCaps* othercaps);

    /* Deinit members */
    ~SampleAlgorithm();

private:
    /* Helper Function to Extract Batch Meta from buffer */
    NvDsBatchMeta* GetNVDS_BatchMeta(GstBuffer* buffer);

    /* Output Processing Thread, push buffer to downstream  */
    void OutputThread(void);

    /* Helper function to Dump NvBufSurface RAW content */
    void DumpNvBufSurface(NvBufSurface* in_surface, NvDsBatchMeta* batch_meta);

    /* Insert Custom Frame */
    void InsertCustomFrame(PacketInfo* packetInfo);

public:
    guint source_id = 0;
    guint m_frameNum = 0;
    gdouble m_scaleFactor = 1.0;
    guint m_frameinsertinterval = 0;
    bool outputthread_stopped = false;

    /* Custom Library Bufferpool */
    GstBufferPool* m_dsBufferPool = NULL;

    /* Output Thread Pointer */
    std::thread* m_outputThread = NULL;

    /* Queue and Lock Management */
    std::queue<PacketInfo> m_processQ;
    std::mutex m_processLock;
    std::condition_variable m_processCV;

    /* Aysnc Stop Handling */
    gboolean m_stop = FALSE;

    /* Vector Containing Key:Value Pair of Custom Lib Properties */
    std::vector<Property> m_vectorProperty;

    void* m_scratchNvBufSurface = NULL;

    // Currently dumps first 5 input video frame into file for demonstration purpose
    // Use vooya or simillar player to view NV12 / RGBA video raw frame
    int dump_max_frames = 5;
};

// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary* CreateCustomAlgoCtx(DSCustom_CreateParams* params)
{
    return new SampleAlgorithm();
}

// Set Init Parameters
bool SampleAlgorithm::SetInitParams(DSCustom_CreateParams* params)
{
    DSCustomLibraryBase::SetInitParams(params);

    BufferPoolConfig pool_config = {0};
    GstStructure* s1 = NULL;

    s1 = gst_caps_get_structure(m_inCaps, 0);

    // Create buffer pool
    // Set the params properly based on this custom library requirement
    pool_config.cuda_mem_type = NVBUF_MEM_CUDA_UNIFIED; //NVBUF_MEM_DEFAULT=0, NVBUF_MEM_CUDA_UNIFIED=3
    pool_config.gpu_id = 0;
    pool_config.max_buffers = 4;
    gst_structure_get_int(s1, "batch-size", &pool_config.batch_size);

    if (pool_config.batch_size == 0)
    {
        // If this component is placed before mux, batch-size value is not set
        // In this case make batch_size = 1
        pool_config.batch_size = 1;
    }

    m_dsBufferPool = CreateBufferPool(&pool_config, m_outCaps);
    if (!m_dsBufferPool)
    {
        throw std::runtime_error("Custom Buffer Pool Creation failed");
    }

    m_outputThread = new std::thread(&SampleAlgorithm::OutputThread, this);

    return true;
}

// Return Compatible Output Caps based on input caps
GstCaps* SampleAlgorithm::GetCompatibleCaps(GstPadDirection direction, GstCaps* in_caps, GstCaps* othercaps)
{
    GstCaps* result = NULL;
    GstStructure *s1, *s2;
    gint width, height;
    gint i, num, denom;
    const gchar* inputFmt = NULL;

    GST_DEBUG(
        "\n----------\ndirection = %d (1=Src, 2=Sink) -> %s:\nCAPS = %s\n",
        direction,
        __func__,
        gst_caps_to_string(in_caps));
    GST_DEBUG("%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

#if 0
  GST_INFO_OBJECT (nvdsvideotemplate, "%s : CAPS = %" GST_PTR_FORMAT "\n\n", __func__, in_caps);
  GST_INFO_OBJECT (nvdsvideotemplate, "%s : OTHER CAPS = %" GST_PTR_FORMAT "\n\n", __func__, othercaps);
#endif

    othercaps = gst_caps_truncate(othercaps);
    othercaps = gst_caps_make_writable(othercaps);

    //num_input_caps = gst_caps_get_size (in_caps);
    int num_output_caps = gst_caps_get_size(othercaps);

    // TODO: Currently it only takes first caps
    s1 = gst_caps_get_structure(in_caps, 0);
    for (i = 0; i < num_output_caps; i++)
    {
        s2 = gst_caps_get_structure(othercaps, i);
        inputFmt = gst_structure_get_string(s1, "format");

        GST_DEBUG("InputFMT = %s \n\n", inputFmt);

        if ((strncmp(inputFmt, FORMAT_RGBA, strlen(FORMAT_RGBA)) != 0))
        {
            g_print("Current version supports RGBA format...\n");
            exit(-1);
        }

        // Check for desired color format
        if ((strncmp(inputFmt, FORMAT_NV12, strlen(FORMAT_NV12)) == 0) ||
            (strncmp(inputFmt, FORMAT_RGBA, strlen(FORMAT_RGBA)) == 0))
        {
            //Set these output caps
            gst_structure_get_int(s1, "width", &width);
            gst_structure_get_int(s1, "height", &height);

            /* otherwise the dimension of the output heatmap needs to be fixated */

            // Here change the width and height on output caps based on the information provided
            // byt the custom library
            gst_structure_fixate_field_nearest_int(s2, "width", width);
            gst_structure_fixate_field_nearest_int(s2, "height", height);
            if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
            {
                gst_structure_fixate_field_nearest_fraction(s2, "framerate", num, denom);
            }

            //gst_structure_remove_fields (s2, "width", "height", "format", NULL);

            // TODO: Get width, height, coloutformat, and framerate from customlibrary API
            // set the new properties accordingly
#if 0
      gst_structure_set (s2, "width", G_TYPE_INT, width * m_scaleFactor,
          "height", G_TYPE_INT, height * m_scaleFactor,
          "format", G_TYPE_STRING, FORMAT_NV12,
          NULL);
#else
            //gst_structure_fixate_field_nearest_double(s2, "width", m_scaleFactor * width);
            //gst_structure_fixate_field_nearest_double(s2, "height", m_scaleFactor * height);
            gst_structure_set(s2, "width", G_TYPE_INT, (gint)(width), NULL);
            gst_structure_set(s2, "height", G_TYPE_INT, (gint)(height), NULL);
            gst_structure_set(s2, "format", G_TYPE_STRING, inputFmt, NULL);
#endif

            result = gst_caps_ref(othercaps);
            gst_caps_unref(othercaps);
            GST_DEBUG("%s : Updated OTHERCAPS = %s \n\n", __func__, gst_caps_to_string(othercaps));
#if 0
  GST_INFO_OBJECT (nvdsvideotemplate, "%s : CAPS = %" GST_PTR_FORMAT "\n\n", __func__, othercaps);
  GST_INFO_OBJECT(nvdsvideotemplate, "CAPS fixate: %" GST_PTR_FORMAT ", direction %d",
      result, direction);
#endif

            break;
        }
        else
        {
            continue;
        }
    }
    return result;
}

bool SampleAlgorithm::HandleEvent(GstEvent* event)
{
    switch (GST_EVENT_TYPE(event))
    {
    case GST_EVENT_EOS:
        m_processLock.lock();
        m_stop = TRUE;
        m_processCV.notify_all();
        m_processLock.unlock();
        while (outputthread_stopped == FALSE)
        {
            //g_print ("waiting for processq to be empty, buffers in processq = %ld\n", m_processQ.size());
            g_usleep(1000);
        }
        break;
    default:
        break;
    }
    if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
    {
        gst_nvevent_parse_stream_eos(event, &source_id);
    }
    return true;
}

// Set Custom Library Specific Properties
bool SampleAlgorithm::SetProperty(Property& prop)
{
    std::cout << "Inside Custom Lib : Setting Prop Key=" << prop.key << " Value=" << prop.value << std::endl;
    m_vectorProperty.emplace_back(prop.key, prop.value);

    try
    {
        if (prop.key.compare("scale-factor") == 0)
        {
            m_scaleFactor = stof(prop.value);
            if (m_scaleFactor == 0 || m_scaleFactor > 20 || m_scaleFactor < 0)
            {
                throw std::out_of_range("out of range scale factor");
            }
        }
        if (prop.key.compare("frame-insert-interval") == 0)
        {
            m_frameinsertinterval = stod(prop.value);
        }
    }
    catch (std::invalid_argument& e)
    {
        std::cout << "Invalid Scale Factor" << std::endl;
        return false;
    }
    catch (std::out_of_range& e)
    {
        std::cout << "Out of Range Scale Factor, provide between 0.0 and 20.0" << std::endl;
        return false;
    }

    return true;
}

/* Deinitialize the Custom Lib context */
SampleAlgorithm::~SampleAlgorithm()
{
    std::unique_lock<std::mutex> lk(m_processLock);
    //std::cout << "Process Q Empty : " << m_processQ.empty() << std::endl;
    m_processCV.wait(lk, [&] { return m_processQ.empty(); });
    m_stop = TRUE;
    m_processCV.notify_all();
    lk.unlock();

    /* Wait for OutputThread to complete */
    if (m_outputThread)
    {
        m_outputThread->join();
    }

    if (m_dsBufferPool)
    {
        gst_buffer_pool_set_active(m_dsBufferPool, FALSE);
        gst_object_unref(m_dsBufferPool);
        m_dsBufferPool = NULL;
    }

    if (m_scratchNvBufSurface)
    {
        cudaFree(&m_scratchNvBufSurface);
        m_scratchNvBufSurface = NULL;
    }
}

// Returns NvDsBatchMeta if present in the gstreamer buffer else NULL
NvDsBatchMeta* SampleAlgorithm::GetNVDS_BatchMeta(GstBuffer* buffer)
{
    gpointer state = NULL;
    GstMeta* gst_meta = NULL;
    NvDsBatchMeta* batch_meta = NULL;

    while ((gst_meta = gst_buffer_iterate_meta(buffer, &state)))
    {
        if (!gst_meta_api_type_has_tag(gst_meta->info->api, _dsmeta_quark))
        {
            continue;
        }
        NvDsMeta* dsmeta = (NvDsMeta*)gst_meta;

        if (dsmeta->meta_type == NVDS_BATCH_GST_META)
        {
            if (batch_meta != NULL)
            {
                GST_WARNING("Multiple NvDsBatchMeta found on buffer %p", buffer);
            }
            batch_meta = (NvDsBatchMeta*)dsmeta->meta_data;
        }
    }

    return batch_meta;
}

/* Process Buffer */
BufferResult SampleAlgorithm::ProcessBuffer(GstBuffer* inbuf)
{
    GstMapInfo in_map_info;
    NvBufSurface* in_surf;
    //BuffferResult result;
    NvDsBatchMeta* batch_meta = NULL;
    //int num_filled = 0;

    GST_DEBUG("CustomLib: ---> Inside %s frame_num = %d\n", __func__, m_frameNum++);

    // TODO: End of Stream Handling
    memset(&in_map_info, 0, sizeof(in_map_info));

    /* Map the buffer contents and get the pointer to NvBufSurface. */
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ))
    {
        GST_ELEMENT_ERROR(
            m_element,
            STREAM,
            FAILED,
            ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__),
            (NULL));
        return BufferResult::Buffer_Error;
    }

    // Assuming that the plugin uses DS NvBufSurface data structure
    in_surf = (NvBufSurface*)in_map_info.data;

    gst_buffer_unmap(inbuf, &in_map_info);

    //batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
    batch_meta = GetNVDS_BatchMeta(inbuf);

#if 0
  // Batch meta is attached when nvstreammux plugin is present in the pipeline and
  // it should be upstream plugin in the pipeline
  if (batch_meta) {
    num_filled = batch_meta->num_frames_in_batch;
  } else {
    num_filled = 1;
  }

  GST_DEBUG ("\tCustomLib: Element : %s in_surf = %p\n", GST_ELEMENT_NAME(m_element), in_surf);
  GST_DEBUG ("\tCustomLib: Num of frames in batch: %d\n", num_filled);
#endif

    // Push buffer to process thread for further processing
    PacketInfo packetInfo;
    packetInfo.inbuf = inbuf;
    packetInfo.frame_num = m_frameNum;

    // Add custom preprocessing logic if required, here
    // Pass the buffer to output_loop for further processing and pusing to next component
    // Currently its just dumping few decoded video frames

    // Enable for dumping the input frame, for debugging purpose
    if (0)
        DumpNvBufSurface(in_surf, batch_meta);

    m_processLock.lock();
    m_processQ.push(packetInfo);
    m_processCV.notify_all();
    m_processLock.unlock();

    return BufferResult::Buffer_Async; //BufferResult::Buffer_Ok;
}

/* Output Processing Thread */
void SampleAlgorithm::OutputThread(void)
{
    GstFlowReturn flow_ret;
    GstBuffer* outBuffer = NULL;
    NvBufSurface* outSurf = NULL;
    std::unique_lock<std::mutex> lk(m_processLock);
    /* Run till signalled to stop. */
    while (1)
    {
        /* Wait if processing queue is empty. */
        if (m_processQ.empty())
        {
            if (m_stop == TRUE)
            {
                break;
            }
            m_processCV.wait(lk);
            continue;
        }

        PacketInfo packetInfo = m_processQ.front();
        m_processQ.pop();

        m_processCV.notify_all();
        lk.unlock();

        // Add custom algorithm logic here
        // Once buffer processing is done, push the buffer to the downstream by using gst_pad_push function

        NvBufSurface* in_surf = getNvBufSurface(packetInfo.inbuf);

        /* Insert custom buffer every after 10 frames */
        if (m_frameinsertinterval)
        {
            if ((packetInfo.frame_num % m_frameinsertinterval) == 0)
            {
                InsertCustomFrame(&packetInfo);
            }
        }

        // Transform mode, hence transform input buffer to output buffer
        GstBuffer* newGstOutBuf = NULL;
        GstFlowReturn result = GST_FLOW_OK;
        result = gst_buffer_pool_acquire_buffer(m_dsBufferPool, &newGstOutBuf, NULL);
        if (result != GST_FLOW_OK)
        {
            GST_ERROR_OBJECT(m_element, "InsertCustomFrame failed error = %d, exiting...", result);
            exit(-1);
        }
        // TODO: Copy meta and transform if required
        if (!gst_buffer_copy_into(newGstOutBuf, packetInfo.inbuf, GST_BUFFER_COPY_META, 0, -1))
        {
            GST_DEBUG("Buffer metadata copy failed \n");
        }
        // Copy previous buffer to new buffer, repreat the frame
        NvBufSurface* out_surf = getNvBufSurface(newGstOutBuf);
        if (!in_surf || !out_surf)
        {
            g_print("CustomLib: NvBufSurface not found in the buffer...exiting...\n");
            exit(-1);
        }

        int b_siz;


        if (!in_surf->batchSize > 1)
        {
            g_print("Batch size greater than 1...\n");
            exit(-1);
        }

        out_surf->numFilled = in_surf->numFilled;

        simple_kernel(in_surf, out_surf);

        outSurf = out_surf;
        outBuffer = newGstOutBuf;

        GST_BUFFER_PTS(outBuffer) = GST_BUFFER_PTS(packetInfo.inbuf);
        // Unref the input buffer
        gst_buffer_unref(packetInfo.inbuf);

        // Output buffer parameters checking
        if (outSurf->numFilled != 0)
        {
            g_assert((guint)m_outVideoInfo.width == outSurf->surfaceList->width);
            g_assert((guint)m_outVideoInfo.height == outSurf->surfaceList->height);
        }

        flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(m_element), outBuffer);
        GST_DEBUG(
            "CustomLib: %s in_surf=%p, Pushing Frame %d to downstream... flow_ret = %d TS=%" GST_TIME_FORMAT " \n",
            __func__,
            in_surf,
            packetInfo.frame_num,
            flow_ret,
            GST_TIME_ARGS(GST_BUFFER_PTS(outBuffer)));

        lk.lock();
        continue;
    }
    outputthread_stopped = true;
    lk.unlock();
    return;
}

// Insert Custom Frame
void SampleAlgorithm::InsertCustomFrame(PacketInfo* packetInfo)
{
    GstBuffer* newGstOutBuf = NULL;
    GstFlowReturn result = GST_FLOW_OK;

    result = gst_buffer_pool_acquire_buffer(m_dsBufferPool, &newGstOutBuf, NULL);
    if (result != GST_FLOW_OK)
    {
        GST_ERROR_OBJECT(m_element, "InsertCustomFrame failed error = %d, exiting...", result);
        exit(-1);
    }
    // TODO: Copy meta and transform if required
    if (!gst_buffer_copy_into(newGstOutBuf, packetInfo->inbuf, GST_BUFFER_COPY_META, 0, -1))
    {
        GST_DEBUG("Buffer metadata copy failed \n");
    }

    // Copy previous buffer to new buffer, repreat the frame
    NvBufSurface* in_surf = getNvBufSurface(packetInfo->inbuf);
    NvBufSurface* out_surf = getNvBufSurface(newGstOutBuf);
    if (!in_surf || !out_surf)
    {
        g_print("CustomLib: NvBufSurface not found in the buffer...exiting...\n");
        exit(-1);
    }
    // Enable below code to copy the frame, else it will insert GREEN frame
    if (0)
    {
        NvBufSurfaceCopy(in_surf, out_surf);
    }

    out_surf->numFilled = in_surf->numFilled;

    // TODO: Do Timestamp management,
    // currently setting is 10ms lesser than next buffer for this newly inserting frame
    GST_BUFFER_PTS(newGstOutBuf) = GST_BUFFER_PTS(packetInfo->inbuf) - 10000000;

    // Increment frame count
    m_frameNum++;

    // Check the surface we are psuhing downstream is of expected width and height values
    // else assert
    g_assert((guint)m_outVideoInfo.width == out_surf->surfaceList->width);
    g_assert((guint)m_outVideoInfo.height == out_surf->surfaceList->height);

    GST_DEBUG(
        "CustomLib: --- Pushed New Custom Buffer %s : InBuf_TS=%" GST_TIME_FORMAT
        " GstOutBuf=%p CustomFrame TS=%" GST_TIME_FORMAT " \n",
        __func__,
        GST_TIME_ARGS(GST_BUFFER_PTS(packetInfo->inbuf)),
        newGstOutBuf,
        GST_TIME_ARGS(GST_BUFFER_PTS(newGstOutBuf)));

    result = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(m_element), newGstOutBuf);
    if (result != GST_FLOW_OK)
    {
        GST_ERROR_OBJECT(m_element, "custom buffer pad push failed error = %d\n", result);
        return;
    }
    return;
}

// Helper function to dump the nvbufsurface, used for debugging purpose
void SampleAlgorithm::DumpNvBufSurface(NvBufSurface* in_surface, NvDsBatchMeta* batch_meta)
{
    guint numFilled = in_surface->numFilled;
    guint source_id = 0;
    guint i = 0;
    std::ofstream outfile;
    void* input_data = NULL;
    guint size = 0;

    if (dump_max_frames)
    {
        dump_max_frames--;
    }
    else
    {
        return;
    }

    for (i = 0; i < numFilled; i++)
    {
        if (batch_meta)
        {
            NvDsFrameMeta* frame_meta = nvds_get_nth_frame_meta(batch_meta->frame_meta_list, i);
            source_id = frame_meta->pad_index;
        }

        std::string tmp = "_" + std::to_string(in_surface->surfaceList[i].width) + "x" +
            std::to_string(in_surface->surfaceList[i].height) + "_" + "BS-" + std::to_string(source_id);

        input_data = in_surface->surfaceList[i].dataPtr;
        //input_size = in_surface->surfaceList[i].dataSize;

        switch (m_inVideoFmt)
        {
        case GST_VIDEO_FORMAT_NV12:
        {
            std::string fname = GST_ELEMENT_NAME(m_element) + tmp + ".nv12";

            size = (in_surface->surfaceList[i].width * in_surface->surfaceList[i].height * 3) / 2;

            if (m_scratchNvBufSurface == NULL)
            {
                ck(cudaMallocHost(&m_scratchNvBufSurface, size));
                outfile.open(fname, std::ofstream::out);
            }
            else
            {
                outfile.open(fname, std::ofstream::out | std::ofstream::app);
            }

            cudaMemcpy2D(
                m_scratchNvBufSurface,
                in_surface->surfaceList[i].width,
                input_data,
                in_surface->surfaceList[i].width,
                in_surface->surfaceList[i].width,
                (in_surface->surfaceList[i].height * 3) / 2,
                cudaMemcpyDeviceToHost);

            outfile.write(reinterpret_cast<char*>(m_scratchNvBufSurface), size);
            outfile.close();
        }
        break;

        case GST_VIDEO_FORMAT_RGBA:
        case GST_VIDEO_FORMAT_BGRx:
        {
            std::string fname = GST_ELEMENT_NAME(m_element) + tmp;

            if (m_inVideoFmt == GST_VIDEO_FORMAT_RGBA)
                fname.append(".rgba");
            else if (m_inVideoFmt == GST_VIDEO_FORMAT_BGRx)
                fname.append(".bgrx");

            size = in_surface->surfaceList[i].width * in_surface->surfaceList[i].height * 4;

            if (m_scratchNvBufSurface == NULL)
            {
                ck(cudaMallocHost(&m_scratchNvBufSurface, size));
                outfile.open(fname, std::ofstream::out);
            }
            else
            {
                outfile.open(fname, std::ofstream::out | std::ofstream::app);
            }

            cudaMemcpy2D(
                m_scratchNvBufSurface,
                in_surface->surfaceList[i].width * 4,
                input_data,
                in_surface->surfaceList[i].width * 4,
                in_surface->surfaceList[i].width * 4,
                in_surface->surfaceList[i].height,
                cudaMemcpyDeviceToHost);

            outfile.write(reinterpret_cast<char*>(m_scratchNvBufSurface), size);
            outfile.close();
        }
        break;

        default:
            g_print("CustomLib: %s : NOT SUPPORTED FORMAT\n", GST_ELEMENT_NAME(m_element));
            break;
        }
    }
}
