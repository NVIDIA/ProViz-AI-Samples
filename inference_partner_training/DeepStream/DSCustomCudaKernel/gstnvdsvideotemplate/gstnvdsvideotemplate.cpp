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

#include <string.h>
#include <sys/time.h>

#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <memory>

#include "gst-nvevent.h"
#include "gstnvdsvideotemplate.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvdsvideotemplate_debug);
#define GST_CAT_DEFAULT gst_nvdsvideotemplate_debug

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_CUSTOMLIB_NAME,
  PROP_GPU_DEVICE_ID,
  PROP_CUSTOMLIB_PROPS,
};

/* Default values for properties */
#define DEFAULT_GPU_ID 0

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvdsvideotemplate_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA }") ";"
        GST_VIDEO_CAPS_MAKE ("{ " "NV12, RGBA }")));

static GstStaticPadTemplate gst_nvdsvideotemplate_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }") ";"));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvdsvideotemplate_parent_class parent_class
G_DEFINE_TYPE (GstNvDsVideoTemplate, gst_nvdsvideotemplate, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdsvideotemplate_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsvideotemplate_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean
gst_nvdsvideotemplate_sink_event (GstBaseTransform * btrans, GstEvent *event);

static gboolean gst_nvdsvideotemplate_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_nvdsvideotemplate_start (GstBaseTransform * btrans);
static gboolean gst_nvdsvideotemplate_stop (GstBaseTransform * btrans);

static GstFlowReturn
gst_nvdsvideotemplate_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf);
static GstFlowReturn
gst_nvdsvideotemplate_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);



/* fixate the caps on the other side */
static GstCaps* gst_nvdsvideotemplate_fixate_caps(GstBaseTransform* btrans,
    GstPadDirection direction, GstCaps* caps, GstCaps* othercaps)
{
  GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (btrans);

  return nvdsvideotemplate->algo_ctx->GetCompatibleCaps(direction, caps, othercaps);
}

static GstCaps *
gst_nvdsvideotemplate_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  //GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (trans);
  GstCaps *ret = gst_caps_copy (caps);

  GST_DEBUG ("Inside Transform_Caps \ncaps = %s\n", gst_caps_to_string(caps));
  GST_DEBUG ("filter_caps = %s\n\n", gst_caps_to_string(filter));

  if (!ret)
    return nullptr;

  if (filter) {
    GstCaps *tmp = gst_caps_intersect (ret, filter);
    GST_DEBUG ("intersect_caps = %s\n\n", gst_caps_to_string(tmp));
    gst_caps_unref (ret);
    return tmp;
  }

  if (0)
  {
    GstCaps *new_caps = NULL;
    GstCapsFeatures *feature = NULL;
    if (direction == GST_PAD_SINK) {
      // TODO : Call custom lib function for caps
      GST_DEBUG ("\nDirection SINK PAD\n");
      new_caps =
          gst_caps_new_simple ("video/x-raw",
          "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
          "height", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
      feature = gst_caps_features_new ("memory:NVMM", NULL);
      gst_caps_set_features (new_caps, 0, feature);
    }
    else if (direction == GST_PAD_SRC) {
      GST_DEBUG ("\nDirection SRC PAD\n");
      new_caps =
          gst_caps_new_simple ("video/x-raw",
          "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
          "height", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
      feature = gst_caps_features_new ("memory:NVMM", NULL);
      gst_caps_set_features (new_caps, 0, feature);
    }

    return new_caps;
  }

  return ret;
}

static gboolean gst_nvdsvideotemplate_accept_caps (GstBaseTransform * btrans, GstPadDirection direction,
        GstCaps * caps)
{
  gboolean ret = TRUE;
  GstNvDsVideoTemplate *space = NULL;
  GstCaps *allowed = NULL;
  GstCapsFeatures *features;

  space = GST_NVDSVIDEOTEMPLATE (btrans);

  GST_DEBUG_OBJECT (btrans, "accept caps %" GST_PTR_FORMAT, caps);

  /* get all the formats we can handle on this pad */
  if (direction == GST_PAD_SINK)
    allowed = space->sinkcaps;
  else
    allowed = space->srccaps;

  if (!allowed) {
    GST_DEBUG_OBJECT (btrans, "failed to get allowed caps");
    goto no_transform_possible;
  }

  features = gst_caps_get_features (caps, 0);
  if (!gst_caps_features_contains (features, GST_CAPS_FEATURE_MEMORY_NVMM))
  {
      GST_DEBUG_OBJECT (btrans, "failed to find HW memory feature");
      goto no_transform_possible;
  }

  GST_DEBUG_OBJECT (btrans, "allowed caps %" GST_PTR_FORMAT, allowed);

  /* intersect with the requested format */
  ret = gst_caps_is_subset (caps, allowed);
  if (!ret) {
    goto no_transform_possible;
  }

done:
  return ret;

  /* ERRORS */
no_transform_possible:
  {
    GST_DEBUG_OBJECT (btrans,
        "could not transform %" GST_PTR_FORMAT " in anything we support", caps);
    ret = FALSE;
    goto done;
  }
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvdsvideotemplate_class_init (GstNvDsVideoTemplateClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  //gstbasetransform_class->passthrough_on_same_caps = TRUE;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_get_property);

  gstbasetransform_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_transform_caps);

  gstbasetransform_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_fixate_caps);
  gstbasetransform_class->accept_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_accept_caps);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_set_caps);
  gstbasetransform_class->sink_event = GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_sink_event);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_stop);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_nvdsvideotemplate_generate_output);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CUSTOMLIB_NAME,
          g_param_spec_string ("customlib-name", "Custom library name",
            "Set custom library Name to be used",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CUSTOMLIB_PROPS,
          g_param_spec_string ("customlib-props", "Custom Library Properties",
            "Set Custom Library Properties (key:value) string, can be set multiple times,"
            "vector is maintained internally",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdsvideotemplate_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdsvideotemplate_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "NvDsVideoTemplate plugin for Transform/In-Place use-cases",
      "NvDsVideoTemplate Plugin for Transform/In-Place use-cases",
      "A custom algorithm can be hooked for Transform/In-Place use-cases",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvdsvideotemplate_init (GstNvDsVideoTemplate * nvdsvideotemplate)
{
#if 0
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvdsvideotemplate);
  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);
#endif
  /* Initialize all property variables to default values */
  nvdsvideotemplate->gpu_id = DEFAULT_GPU_ID;

  nvdsvideotemplate->sinkcaps =
      gst_static_pad_template_get_caps (&gst_nvdsvideotemplate_sink_template);
  nvdsvideotemplate->srccaps =
      gst_static_pad_template_get_caps (&gst_nvdsvideotemplate_src_template);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvdsvideotemplate_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (object);
  switch (prop_id) {
    case PROP_GPU_DEVICE_ID:
      nvdsvideotemplate->gpu_id = g_value_get_uint (value);
      break;
    case PROP_CUSTOMLIB_NAME:
      if (nvdsvideotemplate->custom_lib_name) {
        g_free(nvdsvideotemplate->custom_lib_name);
      }
      nvdsvideotemplate->custom_lib_name = (gchar *)g_value_dup_string (value);
      break;
    case PROP_CUSTOMLIB_PROPS:
    {
      if (!nvdsvideotemplate->vecProp)
      {
        nvdsvideotemplate->vecProp = new std::vector<Property>;
      }
      {
        if (nvdsvideotemplate->custom_prop_string) {
          g_free(nvdsvideotemplate->custom_prop_string);
          nvdsvideotemplate->custom_prop_string = NULL;
        }
        nvdsvideotemplate->custom_prop_string = (gchar *)g_value_dup_string (value);
        std::string propStr (nvdsvideotemplate->custom_prop_string);
        std::size_t found = 0;
        std::size_t start = 0;

        found = propStr.find_first_of(":");
        if (found == 0)
        {
          g_print ("Custom Library property format is invalid, e.g. expected format "
            "requires key and value string seperated "
            " by : i.e. customlib-props=\"[key:value]\"");
          exit(-1);
        }
        nvdsvideotemplate->vecProp->emplace_back (
          propStr.substr(start, found), propStr.substr(found+1));
      }
    }
    break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_nvdsvideotemplate_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (object);

  switch (prop_id) {
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvdsvideotemplate->gpu_id);
      break;
    case PROP_CUSTOMLIB_NAME:
      g_value_set_string (value, nvdsvideotemplate->custom_lib_name);
      break;
    case PROP_CUSTOMLIB_PROPS:
      g_value_set_string (value, nvdsvideotemplate->custom_prop_string);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the process thread
 */
static gboolean
gst_nvdsvideotemplate_start (GstBaseTransform * btrans)
{
  GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (btrans);
  std::string nvtx_str("GstNvDsVideoTemplate ");
  bool ret;

  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  CHECK_CUDA_STATUS (cudaSetDevice (nvdsvideotemplate->gpu_id),
      "Unable to set cuda device");

  nvdsvideotemplate->nvtx_domain = nvtx_domain_ptr.release ();

  try
  {
      nvdsvideotemplate->algo_factory = new DSCustomLibrary_Factory();
      nvdsvideotemplate->algo_ctx = nvdsvideotemplate->algo_factory->CreateCustomAlgoCtx(nvdsvideotemplate->custom_lib_name);

      if(nvdsvideotemplate->vecProp && nvdsvideotemplate->vecProp->size()) {
          std::cout << "Setting custom lib properties # " << nvdsvideotemplate->vecProp->size() << std::endl;
          for(std::vector<Property>::iterator it = nvdsvideotemplate->vecProp->begin(); it != nvdsvideotemplate->vecProp->end(); ++it) {
              std::cout << "Adding Prop: " << it->key << " : " << it->value << std::endl;
              ret = nvdsvideotemplate->algo_ctx->SetProperty(*it);
              if (!ret) {
                  goto error;
              }
          }
      }
  }
  catch (const std::runtime_error& e)
  {
      std::cout << e.what() << "\n";
      return FALSE;
  }
  catch (...)
  {
    std::cout << "caught exception" << std::endl;
    return FALSE;
  }

  return TRUE;

error:
  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_nvdsvideotemplate_stop (GstBaseTransform * btrans)
{
  GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (btrans);

  nvdsvideotemplate->stop = TRUE;

  if (nvdsvideotemplate->algo_ctx)
    delete nvdsvideotemplate->algo_ctx;

  if (nvdsvideotemplate->algo_factory)
    delete nvdsvideotemplate->algo_factory;

  if (nvdsvideotemplate->vecProp)
    delete nvdsvideotemplate->vecProp;

  if (nvdsvideotemplate->custom_lib_name) {
    g_free(nvdsvideotemplate->custom_lib_name);
    nvdsvideotemplate->custom_lib_name = NULL;
  }
  if (nvdsvideotemplate->custom_prop_string) {
    g_free(nvdsvideotemplate->custom_prop_string);
    nvdsvideotemplate->custom_prop_string = NULL;
  }

  GST_DEBUG_OBJECT (nvdsvideotemplate, "ctx lib released \n");
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvdsvideotemplate_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (btrans);
  DSCustom_CreateParams params = {0};

  /* Save the input & output video information, since this will be required later. */
  gst_video_info_from_caps (&nvdsvideotemplate->in_video_info, incaps);
  gst_video_info_from_caps (&nvdsvideotemplate->out_video_info, outcaps);

  CHECK_CUDA_STATUS (cudaSetDevice (nvdsvideotemplate->gpu_id),
      "Unable to set cuda device");

  // TODO: Manage the ctx in case of DRC like cases here
  params.m_element = btrans;
  params.m_inCaps = incaps;
  params.m_outCaps = outcaps;
  GST_DEBUG("outcaps received = %s\n", gst_caps_to_string (outcaps));
  params.m_gpuId = nvdsvideotemplate->gpu_id;

  return nvdsvideotemplate->algo_ctx->SetInitParams(&params);

error:
  return FALSE;
}

static gboolean
gst_nvdsvideotemplate_sink_event (GstBaseTransform * btrans, GstEvent *event)
{
    gboolean ret = TRUE;
    GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (btrans);
    guint source_id = 0;

    switch (GST_EVENT_TYPE(event))
    {
        case GST_EVENT_EOS:
            g_print ("EOS received\n");
            ret  = nvdsvideotemplate->algo_ctx->HandleEvent(event);
            if (!ret)
                return ret;
            break;
        default:
            break;

    }

    if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
    {
        gst_nvevent_parse_stream_eos (event, &source_id);
    }

    return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event(btrans, event);
}


/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvdsvideotemplate_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstFlowReturn flow_ret;
  GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (btrans);

  BufferResult result = BufferResult::Buffer_Async;
  GST_DEBUG("nvdsvideotemplate: Inside %s \n", __func__);

  // Call the callback of user provided library
  // check the return type
  // based on return type push the buffer or just return as use is going to handle pushing of data
  if (nvdsvideotemplate->algo_ctx)
  {
    result = nvdsvideotemplate->algo_ctx->ProcessBuffer (inbuf);

    if (result == BufferResult::Buffer_Ok) {
      flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvdsvideotemplate), inbuf);
      GST_DEBUG("nvdsvideotemplate: -- Forwarding Buffer to downstream, flow_ret = %d\n", flow_ret);
      return flow_ret;
    } else if (result == BufferResult::Buffer_Drop) {
        GST_DEBUG("nvdsvideotemplate: -- Dropping Buffer");
        // TODO unref the buffer so that it will be dropped
        return GST_FLOW_OK;
    } else if (result == BufferResult::Buffer_Error) {
        GST_DEBUG ("nvdsvideotemplate: -- Buffer_Error Buffer");
        return GST_FLOW_ERROR;
    } else if (result == BufferResult::Buffer_Async) {
        GST_DEBUG ("nvdsvideotemplate: -- Buffer_Async Received, custom lib to push the Buffer to downstream\n");
        return GST_FLOW_OK;
    }
  }

  flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvdsvideotemplate), inbuf);
  GST_DEBUG ("Example2Plugin: -- Sending Buffer to downstream, flow_ret = %d\n", flow_ret);
  return GST_FLOW_OK;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_nvdsvideotemplate_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  //GstNvDsVideoTemplate *nvdsvideotemplate = GST_NVDSVIDEOTEMPLATE (btrans);
  //GST_DEBUG("nvdsvideotemplate: Inside %s\n", __func__);
  return GST_FLOW_OK;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdsvideotemplate_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdsvideotemplate_debug, "nvdsvideotemplate", 0,
      "nvdsvideotemplate plugin");

  return gst_element_register (plugin, "nvdsvideotemplate", GST_RANK_PRIMARY,
      GST_TYPE_NVDSVIDEOTEMPLATE);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_videotemplate,
    DESCRIPTION, nvdsvideotemplate_plugin_init, "5.1", LICENSE, BINARY_PACKAGE,
    URL)
