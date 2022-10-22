/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This example demonstrates how to use the CUDA Direct3D bindings to
 * transfer data between CUDA and DX9 2D, CubeMap, and Volume Textures.
 */

#pragma warning(disable : 4312)

#include <windows.h>
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <channel_descriptor.h>
#include <driver_functions.h>
#include <d3dcompiler.h>

// includes, project
#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

#include <dxgi1_2.h> // Required for using External Resource Interoperability

#define MAX_EPSILON 10

static char *SDK_name = "simpleD3D11Texture";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter *g_pCudaCapableAdapter = NULL;  // Adapter to use
ID3D11Device *g_pd3dDevice = NULL;           // Our rendering device
ID3D11DeviceContext *g_pd3dDeviceContext = NULL;
IDXGISwapChain *g_pSwapChain = NULL;  // The swap chain of the window
ID3D11RenderTargetView *g_pSwapChainRTV =
    NULL;  // The Render target view on the swap chain ( used for clear)
ID3D11RasterizerState *g_pRasterState = NULL;

#ifdef USEEFFECT
#pragma message( \
    ">>>> NOTE : Using Effect library (see DXSDK Utility folder for sources)")
#pragma message( \
    ">>>> WARNING : Currently only libs for vc9 are provided with the sample. See DXSDK for more...")
#pragma message( \
    ">>>> WARNING : The effect is currently failing... some strange internal error in Effect lib")
ID3DX11Effect *g_pSimpleEffect = NULL;
ID3DX11EffectTechnique *g_pSimpleTechnique = NULL;
ID3DX11EffectVectorVariable *g_pvQuadRect = NULL;
ID3DX11EffectScalarVariable *g_pUseCase = NULL;
ID3DX11EffectShaderResourceVariable *g_pTexture2D = NULL;
ID3DX11EffectShaderResourceVariable *g_pTexture3D = NULL;
ID3DX11EffectShaderResourceVariable *g_pTextureCube = NULL;

static const char g_simpleEffectSrc[] =
    "float4 g_vQuadRect; \n"
    "int g_UseCase; \n"
    "Texture2D g_Texture2D; \n"
    "Texture3D g_Texture3D; \n"
    "TextureCube g_TextureCube; \n"
    "\n"
    "SamplerState samLinear{ \n"
    "    Filter = MIN_MAG_LINEAR_MIP_POINT; \n"
    "};\n"
    "\n"
    "struct Fragment{ \n"
    "    float4 Pos : SV_POSITION;\n"
    "    float3 Tex : TEXCOORD0; };\n"
    "\n"
    "Fragment VS( uint vertexId : SV_VertexID )\n"
    "{\n"
    "    Fragment f;\n"
    "    f.Tex = float3( 0.f, 0.f, 0.f); \n"
    "    if (vertexId == 1) f.Tex.x = 1.f; \n"
    "    else if (vertexId == 2) f.Tex.y = 1.f; \n"
    "    else if (vertexId == 3) f.Tex.xy = float2(1.f, 1.f); \n"
    "    \n"
    "    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1);\n"
    "    \n"
    "    if (g_UseCase == 1) { \n"
    "        if (vertexId == 1) f.Tex.z = 0.5f; \n"
    "        else if (vertexId == 2) f.Tex.z = 0.5f; \n"
    "        else if (vertexId == 3) f.Tex.z = 1.f; \n"
    "    } \n"
    "    else if (g_UseCase >= 2) { \n"
    "        f.Tex.xy = f.Tex.xy * 2.f - 1.f; \n"
    "    } \n"
    "    return f;\n"
    "}\n"
    "\n"
    "float4 PS( Fragment f ) : SV_Target\n"
    "{\n"
    "    if (g_UseCase == 0) return g_Texture2D.Sample( samLinear, f.Tex.xy ); "
    "\n"
    "    else if (g_UseCase == 1) return g_Texture3D.Sample( samLinear, f.Tex "
    "); \n"
    "    else if (g_UseCase == 2) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.xy, 1.0) ); \n"
    "    else if (g_UseCase == 3) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.xy, -1.0) ); \n"
    "    else if (g_UseCase == 4) return g_TextureCube.Sample( samLinear, "
    "float3(1.0, f.Tex.xy) ); \n"
    "    else if (g_UseCase == 5) return g_TextureCube.Sample( samLinear, "
    "float3(-1.0, f.Tex.xy) ); \n"
    "    else if (g_UseCase == 6) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.x, 1.0, f.Tex.y) ); \n"
    "    else if (g_UseCase == 7) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.x, -1.0, f.Tex.y) ); \n"
    "    else return float4(f.Tex, 1);\n"
    "}\n"
    "\n"
    "technique11 Render\n"
    "{\n"
    "    pass P0\n"
    "    {\n"
    "        SetVertexShader( CompileShader( vs_5_0, VS() ) );\n"
    "        SetGeometryShader( NULL );\n"
    "        SetPixelShader( CompileShader( ps_5_0, PS() ) );\n"
    "    }\n"
    "}\n"
    "\n";
#else
//
// Vertex and Pixel shaders here : VS() & PS()
//
static const char g_simpleShaders[] =
    "cbuffer cbuf \n"
    "{ \n"
    "  float4 g_vQuadRect; \n"
    "  int g_UseCase; \n"
    "} \n"
    "Texture2D g_Texture2D; \n"
    "Texture3D g_Texture3D; \n"
    "TextureCube g_TextureCube; \n"
    "\n"
    "SamplerState samLinear{ \n"
    "    Filter = MIN_MAG_LINEAR_MIP_POINT; \n"
    "};\n"
    "\n"
    "struct Fragment{ \n"
    "    float4 Pos : SV_POSITION;\n"
    "    float3 Tex : TEXCOORD0; };\n"
    "\n"
    "Fragment VS( uint vertexId : SV_VertexID )\n"
    "{\n"
    "    Fragment f;\n"
    "    f.Tex = float3( 0.f, 0.f, 0.f); \n"
    "    if (vertexId == 1) f.Tex.x = 1.f; \n"
    "    else if (vertexId == 2) f.Tex.y = 1.f; \n"
    "    else if (vertexId == 3) f.Tex.xy = float2(1.f, 1.f); \n"
    "    \n"
    "    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1);\n"
    "    \n"
    "    if (g_UseCase == 1) { \n"
    "        if (vertexId == 1) f.Tex.z = 0.5f; \n"
    "        else if (vertexId == 2) f.Tex.z = 0.5f; \n"
    "        else if (vertexId == 3) f.Tex.z = 1.f; \n"
    "    } \n"
    "    else if (g_UseCase >= 2) { \n"
    "        f.Tex.xy = f.Tex.xy * 2.f - 1.f; \n"
    "    } \n"
    "    return f;\n"
    "}\n"
    "\n"
    "float4 PS( Fragment f ) : SV_Target\n"
    "{\n"
    "    if (g_UseCase == 0) return g_Texture2D.Sample( samLinear, f.Tex.xy ); "
    "\n"
    "    else if (g_UseCase == 1) return g_Texture3D.Sample( samLinear, f.Tex "
    "); \n"
    "    else if (g_UseCase == 2) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.xy, 1.0) ); \n"
    "    else if (g_UseCase == 3) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.xy, -1.0) ); \n"
    "    else if (g_UseCase == 4) return g_TextureCube.Sample( samLinear, "
    "float3(1.0, f.Tex.xy) ); \n"
    "    else if (g_UseCase == 5) return g_TextureCube.Sample( samLinear, "
    "float3(-1.0, f.Tex.xy) ); \n"
    "    else if (g_UseCase == 6) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.x, 1.0, f.Tex.y) ); \n"
    "    else if (g_UseCase == 7) return g_TextureCube.Sample( samLinear, "
    "float3(f.Tex.x, -1.0, f.Tex.y) ); \n"
    "    else return float4(f.Tex, 1);\n"
    "}\n"
    "\n";

struct ConstantBuffer {
  float vQuadRect[4];
  int UseCase;
};

ID3D11VertexShader *g_pVertexShader;
ID3D11PixelShader *g_pPixelShader;
ID3D11Buffer *g_pConstantBuffer;
ID3D11SamplerState *g_pSamplerState;

#endif
// testing/tracing function used pervasively in tests.  if the condition is
// unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x)                                                  \
  if (!(x)) {                                                            \
    fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, \
            __FILE__, __LINE__);                                         \
    return 1;                                                            \
  }

bool g_bDone = false;
bool g_bPassed = true;

int *pArgc = NULL;
char **pArgv = NULL;

const unsigned int g_WindowWidth = 720;
const unsigned int g_WindowHeight = 720;

int g_iFrameToCompare = 10;

// Data structure for volume textures shared between DX11 and CUDA
struct Texture3D {
  ID3D11Texture3D *pTexture;
  cudaExternalMemory_t cuda_external_memory = nullptr;
  cudaMipmappedArray* cuda_mip_mapped_array = 0;
  //cudaGraphicsResource *cudaResource;
  size_t pitch;
  int width;
  int height;
  int depth;
#ifndef USEEFFECT
  int offsetInShader;
#endif
};

// The CUDA kernel launchers that get called
extern "C" {
bool cuda_texture_2d(void *surface, size_t width, size_t height, size_t pitch,
                     float t);
bool cuda_texture_3d(void *surface, int width, int height, int depth,
                     size_t pitch, size_t pitchslice, float t);
bool cuda_texture_cube(void *surface, int width, int height, size_t pitch,
                       int face, float t);
}

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd);
HRESULT InitTexture3D(const int width, const int height, const int depth, Texture3D& g_texture_3d);

void CleanupTexture(Texture3D& g_texture_3d);
void CleanupOthers();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#define NAME_LEN 512

bool findCUDADevice() {
  int nGraphicsGPU = 0;
  int deviceCount = 0;
  bool bFoundGraphics = false;
  char devname[NAME_LEN];

  // This function call returns 0 if there are no CUDA capable devices.
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0) {
    printf("> There are no device(s) supporting CUDA\n");
    return false;
  } else {
    printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
  }

  // Get CUDA device properties
  cudaDeviceProp deviceProp;

  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaGetDeviceProperties(&deviceProp, dev);
    STRCPY(devname, NAME_LEN, deviceProp.name);
    printf("> GPU %d: %s\n", dev, devname);
  }

  return true;
}

bool findDXDevice(char *dev_name) {
  HRESULT hr = S_OK;
  cudaError cuStatus;

  // Iterate through the candidate adapters
  IDXGIFactory *pFactory;
  hr = sFnPtr_CreateDXGIFactory(__uuidof(IDXGIFactory), (void **)(&pFactory));

  if (!SUCCEEDED(hr)) {
    printf("> No DXGI Factory created.\n");
    return false;
  }

  UINT adapter = 0;

  for (; !g_pCudaCapableAdapter; ++adapter) {
    // Get a candidate DXGI adapter
    IDXGIAdapter *pAdapter = NULL;
    hr = pFactory->EnumAdapters(adapter, &pAdapter);

    if (FAILED(hr)) {
      break;  // no compatible adapters found
    }

    // Query to see if there exists a corresponding compute device
    int cuDevice;
    cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
    printLastCudaError("cudaD3D11GetDevice failed");  // This prints and resets
                                                      // the cudaError to
                                                      // cudaSuccess

    if (cudaSuccess == cuStatus) {
      // If so, mark it as the one against which to create our d3d10 device
      g_pCudaCapableAdapter = pAdapter;
      g_pCudaCapableAdapter->AddRef();
    }

    pAdapter->Release();
  }

  printf("> Found %d D3D11 Adapater(s).\n", (int)adapter);

  pFactory->Release();

  if (!g_pCudaCapableAdapter) {
    printf("> Found 0 D3D11 Adapater(s) /w Compute capability.\n");
    return false;
  }

  DXGI_ADAPTER_DESC adapterDesc;
  g_pCudaCapableAdapter->GetDesc(&adapterDesc);
  wcstombs(dev_name, adapterDesc.Description, 128);

  printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
  printf("> %s\n", dev_name);

  return true;
}

bool GetTextureBufferSizeByte(const D3D11_TEXTURE3D_DESC& d3d11_texture3d_desc, unsigned long long* buffer_size_byte) {
  const unsigned long long num_total_voxels = (unsigned long long)(d3d11_texture3d_desc.Width) * (unsigned long long)(d3d11_texture3d_desc.Height) * (unsigned long long)(d3d11_texture3d_desc.Depth);
  switch (d3d11_texture3d_desc.Format) {
  case DXGI_FORMAT::DXGI_FORMAT_R8_UINT:
  case DXGI_FORMAT::DXGI_FORMAT_R8_UNORM:
    *buffer_size_byte = num_total_voxels * sizeof(unsigned char);
    return true;
  case DXGI_FORMAT::DXGI_FORMAT_R8_SINT:
  case DXGI_FORMAT::DXGI_FORMAT_R8_SNORM:
    *buffer_size_byte = num_total_voxels * sizeof(signed char);
    return true;
  case DXGI_FORMAT::DXGI_FORMAT_R16_UINT:
  case DXGI_FORMAT::DXGI_FORMAT_R16_UNORM:
    *buffer_size_byte = num_total_voxels * sizeof(unsigned short);
    return true;
  case DXGI_FORMAT::DXGI_FORMAT_R16_SINT:
  case DXGI_FORMAT::DXGI_FORMAT_R16_SNORM:
    *buffer_size_byte = num_total_voxels * sizeof(short);
    return true;
  default:
    std::cout << "Only DXGI_FORMAT_{R8, R16}_{UINT, UNORM, SINT, SNORM} are currently supported." << std::endl;
    *buffer_size_byte = 0;
    return false;
  }
}

bool GetCudaChannelFormatDesc(const DXGI_FORMAT& dxgi_format, cudaChannelFormatDesc* desc) {
  memset(desc, 0, sizeof(desc));
  switch (dxgi_format) {
  case DXGI_FORMAT::DXGI_FORMAT_R8_UINT:
  case DXGI_FORMAT::DXGI_FORMAT_R8_UNORM:
    *desc = cudaCreateChannelDesc<unsigned char>();
    return true;
  case DXGI_FORMAT::DXGI_FORMAT_R8_SINT:
  case DXGI_FORMAT::DXGI_FORMAT_R8_SNORM:
    *desc = cudaCreateChannelDesc<signed char>();
    return true;
  case DXGI_FORMAT::DXGI_FORMAT_R16_UINT:
  case DXGI_FORMAT::DXGI_FORMAT_R16_UNORM:
    *desc = cudaCreateChannelDesc<unsigned short>();
    return true;
  case DXGI_FORMAT::DXGI_FORMAT_R16_SINT:
  case DXGI_FORMAT::DXGI_FORMAT_R16_SNORM:
    *desc = cudaCreateChannelDesc<short>();
    return true;
  default:
    std::cout << "Only DXGI_FORMAT_{R8, R16}_{UINT, UNORM, SINT, SNORM} are currently supported." << std::endl;
    return false;
  }
}

bool InitTexture3DInterop(const int width, const int height, const int depth, Texture3D& g_texture_3d) {
  if (SUCCEEDED(InitTexture3D(width, height, depth, g_texture_3d))) {
    //cudaGraphicsD3D11RegisterResource(&g_texture_3d.cudaResource,
    //  g_texture_3d.pTexture,
    //  cudaGraphicsRegisterFlagsNone);
    //getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_3d) failed");

    D3D11_TEXTURE3D_DESC texture3d_desc;
    g_texture_3d.pTexture->GetDesc(&texture3d_desc);

    unsigned long long buffer_size_byte;
    if (!GetTextureBufferSizeByte(texture3d_desc, &buffer_size_byte)) { return false; }

    IDXGIResource1* dxgi_resource;
    HANDLE d3d11_texture_3d_shared_handle;
    if (FAILED(g_texture_3d.pTexture->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgi_resource))) {
      std::cout << "Error: IDXGIResource1 from D3D11_buffer could not be acquired." << std::endl;
      return false;
    }
    if (FAILED(dxgi_resource->GetSharedHandle(&d3d11_texture_3d_shared_handle))) { // Do not use CloseHandle(shared_handle) for this handle because it is not an NT handle. It causes "An invalid handle was specified." error.
      std::cout << "Error: shared handle could not be acquired." << std::endl;
      dxgi_resource->Release();
      return false;
    }

    cudaExternalMemoryHandleDesc external_memory_handle_desc;
    memset(&external_memory_handle_desc, 0, sizeof(external_memory_handle_desc));
    external_memory_handle_desc.type = cudaExternalMemoryHandleTypeD3D11ResourceKmt;
    external_memory_handle_desc.size = buffer_size_byte;
    external_memory_handle_desc.flags = cudaExternalMemoryDedicated;
    external_memory_handle_desc.handle.win32.handle = (void*)d3d11_texture_3d_shared_handle;
    cudaError_t cuda_error = cudaImportExternalMemory(&(g_texture_3d.cuda_external_memory), &external_memory_handle_desc);
    //getLastCudaError("cudaImportExternalMemory (g_texture_3d) failed");
    if (cuda_error != cudaSuccess) {
      dxgi_resource->Release();
      return false;
    }

    cudaChannelFormatDesc cuda_channel_format_desc;
    if (!GetCudaChannelFormatDesc(texture3d_desc.Format, &cuda_channel_format_desc)) { return false; }

    const unsigned int bytes_per_pixel = buffer_size_byte / ((unsigned long long)texture3d_desc.Width * texture3d_desc.Height * texture3d_desc.Depth);

    cudaExternalMemoryMipmappedArrayDesc external_memory_mipmapped_array_desc;
    memset(&external_memory_mipmapped_array_desc, 0, sizeof(external_memory_mipmapped_array_desc));
    external_memory_mipmapped_array_desc.offset = 0;
    external_memory_mipmapped_array_desc.formatDesc = cuda_channel_format_desc;
    external_memory_mipmapped_array_desc.extent = make_cudaExtent(texture3d_desc.Width/* * bytes_per_pixel*/, texture3d_desc.Height/* * bytes_per_pixel*/, texture3d_desc.Depth/* * bytes_per_pixel*/);
    external_memory_mipmapped_array_desc.flags = 0;
    external_memory_mipmapped_array_desc.numLevels = texture3d_desc.MipLevels;
    cuda_error = cudaExternalMemoryGetMappedMipmappedArray(&(g_texture_3d.cuda_mip_mapped_array), g_texture_3d.cuda_external_memory, &external_memory_mipmapped_array_desc);
    //getLastCudaError("cudaExternalMemoryGetMappedMipmappedArray (g_texture_3d) failed");
    if (cuda_error != cudaSuccess) {
      std::cout << "cudaExternalMemoryGetMappedMipmappedArray (g_texture_3d) failed " << cudaGetErrorName(cuda_error) << " at Texture3D voxels(" << texture3d_desc.Width << ", " << texture3d_desc.Height << ", " << texture3d_desc.Depth << ")" << std::endl;
      dxgi_resource->Release();
      return false;
    }
    else {
      std::cout << "Succeeded at Texture3D voxels(" << texture3d_desc.Width << ", " << texture3d_desc.Height << ", " << texture3d_desc.Depth << ")" << std::endl;
    }
    dxgi_resource->Release();
    return true;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  char device_name[256];
  char *ref_file = NULL;

  pArgc = &argc;
  pArgv = argv;

  printf("[%s] - Starting...\n", SDK_name);

  if (!findCUDADevice())  // Search for CUDA GPU
  {
    printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
    exit(EXIT_SUCCESS);
  }

  if (!dynlinkLoadD3D11API())  // Search for D3D API (locate drivers, does not
                               // mean device is found)
  {
    printf("> D3D11 API libraries NOT found on.. Exiting.\n");
    dynlinkUnloadD3D11API();
    exit(EXIT_SUCCESS);
  }

  if (!findDXDevice(device_name))  // Search for D3D Hardware Device
  {
    printf("> D3D11 Graphics Device NOT found.. Exiting.\n");
    dynlinkUnloadD3D11API();
    exit(EXIT_SUCCESS);
  }

  // command line options
  if (argc > 1) {
    // automatied build testing harness
    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
      getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
  }

//
// create window
//
// Register the window class
#if 1
  WNDCLASSEX wc = {sizeof(WNDCLASSEX),
                   CS_CLASSDC,
                   MsgProc,
                   0L,
                   0L,
                   GetModuleHandle(NULL),
                   NULL,
                   NULL,
                   NULL,
                   NULL,
                   "CUDA SDK",
                   NULL};
  RegisterClassEx(&wc);

  // Create the application's window
  int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
  int yMenu = ::GetSystemMetrics(SM_CYMENU);
  int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
  HWND hWnd = CreateWindow(
      wc.lpszClassName, "CUDA/D3D11 Texture InterOP", WS_OVERLAPPEDWINDOW, 0, 0,
      g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu, NULL,
      NULL, wc.hInstance, NULL);
#else
  static WNDCLASSEX wc = {
      sizeof(WNDCLASSEX),    CS_CLASSDC, MsgProc, 0L,   0L,
      GetModuleHandle(NULL), NULL,       NULL,    NULL, NULL,
      "CudaD3D9Tex",         NULL};
  RegisterClassEx(&wc);
  HWND hWnd = CreateWindow("CudaD3D9Tex", "CUDA D3D9 Texture Interop",
                           WS_OVERLAPPEDWINDOW, 0, 0, 800, 320,
                           GetDesktopWindow(), NULL, wc.hInstance, NULL);
#endif

  ShowWindow(hWnd, SW_SHOWDEFAULT);
  UpdateWindow(hWnd);

  // Initialize Direct3D
  if (SUCCEEDED(InitD3D(hWnd))) {
    std::vector<int> success_depths;
    // 3D
    for (int depth = 1; depth <= 512; ++depth) {
      Texture3D g_texture_3d;
      const int width = 512;
      const int height = 512;
      const bool ret = InitTexture3DInterop(width, height, depth, g_texture_3d);
      if (ret) {
        success_depths.push_back(depth);
      }
      CleanupTexture(g_texture_3d);
    }

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Success depths are ";
    for (const int success_depth : success_depths) {
      std::cout << success_depth << ", ";
    }
    std::cout << std::endl;
  }
  CleanupOthers();
  // Release D3D Library (after message loop)
  dynlinkUnloadD3D11API();

  // Unregister windows class
  UnregisterClass(wc.lpszClassName, wc.hInstance);

  //
  // and exit
  //
  printf("> %s running on %s exiting...\n", SDK_name, device_name);

  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

//-----------------------------------------------------------------------------
// Name: InitD3D()
// Desc: Initializes Direct3D
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd) {
  HRESULT hr = S_OK;

  // Set up the structure used to create the device and swapchain
  DXGI_SWAP_CHAIN_DESC sd;
  ZeroMemory(&sd, sizeof(sd));
  sd.BufferCount = 1;
  sd.BufferDesc.Width = g_WindowWidth;
  sd.BufferDesc.Height = g_WindowHeight;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferDesc.RefreshRate.Numerator = 60;
  sd.BufferDesc.RefreshRate.Denominator = 1;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  sd.OutputWindow = hWnd;
  sd.SampleDesc.Count = 1;
  sd.SampleDesc.Quality = 0;
  sd.Windowed = TRUE;

  D3D_FEATURE_LEVEL tour_fl[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1,
                                 D3D_FEATURE_LEVEL_10_0};
  D3D_FEATURE_LEVEL flRes;
  // Create device and swapchain
  hr = sFnPtr_D3D11CreateDeviceAndSwapChain(
      g_pCudaCapableAdapter,
      D3D_DRIVER_TYPE_UNKNOWN,  // D3D_DRIVER_TYPE_HARDWARE,
      NULL,  // HMODULE Software
      0,  // UINT Flags
      tour_fl,  // D3D_FEATURE_LEVEL* pFeatureLevels
      3,  // FeatureLevels
      D3D11_SDK_VERSION,  // UINT SDKVersion
      &sd,  // DXGI_SWAP_CHAIN_DESC* pSwapChainDesc
      &g_pSwapChain,  // IDXGISwapChain** ppSwapChain
      &g_pd3dDevice,  // ID3D11Device** ppDevice
      &flRes,  // D3D_FEATURE_LEVEL* pFeatureLevel
      &g_pd3dDeviceContext  // ID3D11DeviceContext** ppImmediateContext
      );
  AssertOrQuit(SUCCEEDED(hr));

  g_pCudaCapableAdapter->Release();

  // Get the immediate DeviceContext
  g_pd3dDevice->GetImmediateContext(&g_pd3dDeviceContext);

  // Create a render target view of the swapchain
  ID3D11Texture2D *pBuffer;
  hr =
      g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID *)&pBuffer);
  AssertOrQuit(SUCCEEDED(hr));

  hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, NULL, &g_pSwapChainRTV);
  AssertOrQuit(SUCCEEDED(hr));
  pBuffer->Release();

  g_pd3dDeviceContext->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

  // Setup the viewport
  D3D11_VIEWPORT vp;
  vp.Width = g_WindowWidth;
  vp.Height = g_WindowHeight;
  vp.MinDepth = 0.0f;
  vp.MaxDepth = 1.0f;
  vp.TopLeftX = 0;
  vp.TopLeftY = 0;
  g_pd3dDeviceContext->RSSetViewports(1, &vp);

#ifdef USEEFFECT
  // Setup the effect
  {
    ID3D10Blob *effectCode, *effectErrors;
    hr = D3DX11CompileFromMemory(
        g_simpleEffectSrc, sizeof(g_simpleEffectSrc), "NoFile", NULL, NULL, "",
        "fx_5_0",
        D3D10_SHADER_OPTIMIZATION_LEVEL0 |
            D3D10_SHADER_ENABLE_BACKWARDS_COMPATIBILITY | D3D10_SHADER_DEBUG,
        0, 0, &effectCode, &effectErrors, 0);

    if (FAILED(hr)) {
      const char *pStr = (const char *)effectErrors->GetBufferPointer();
      printf(pStr);
      assert(1);
    }

    hr = D3DX11CreateEffectFromMemory(
        effectCode->GetBufferPointer(), effectCode->GetBufferSize(),
        0 /*FXFlags*/, g_pd3dDevice, &g_pSimpleEffect);
    AssertOrQuit(SUCCEEDED(hr));
    g_pSimpleTechnique = g_pSimpleEffect->GetTechniqueByName("Render");

    g_pvQuadRect =
        g_pSimpleEffect->GetVariableByName("g_vQuadRect")->AsVector();
    g_pUseCase = g_pSimpleEffect->GetVariableByName("g_UseCase")->AsScalar();

    g_pTexture2D =
        g_pSimpleEffect->GetVariableByName("g_Texture2D")->AsShaderResource();
    g_pTexture3D =
        g_pSimpleEffect->GetVariableByName("g_Texture3D")->AsShaderResource();
    g_pTextureCube =
        g_pSimpleEffect->GetVariableByName("g_TextureCube")->AsShaderResource();
  }
#else
  ID3DBlob *pShader;
  ID3DBlob *pErrorMsgs;
  // Vertex shader
  {
    hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL,
                    NULL, "VS", "vs_4_0", 0 /*Flags1*/, 0 /*Flags2*/, &pShader,
                    &pErrorMsgs);

    if (FAILED(hr)) {
      const char *pStr = (const char *)pErrorMsgs->GetBufferPointer();
      printf(pStr);
    }

    AssertOrQuit(SUCCEEDED(hr));
    hr = g_pd3dDevice->CreateVertexShader(pShader->GetBufferPointer(),
                                          pShader->GetBufferSize(), NULL,
                                          &g_pVertexShader);
    AssertOrQuit(SUCCEEDED(hr));
    // Let's bind it now : no other vtx shader will replace it...
    g_pd3dDeviceContext->VSSetShader(g_pVertexShader, NULL, 0);
    // hr = g_pd3dDevice->CreateInputLayout(...pShader used for signature...) No
    // need
  }
  // Pixel shader
  {
    hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL,
                    NULL, "PS", "ps_4_0", 0 /*Flags1*/, 0 /*Flags2*/, &pShader,
                    &pErrorMsgs);

    AssertOrQuit(SUCCEEDED(hr));
    hr = g_pd3dDevice->CreatePixelShader(pShader->GetBufferPointer(),
                                         pShader->GetBufferSize(), NULL,
                                         &g_pPixelShader);
    AssertOrQuit(SUCCEEDED(hr));
    // Let's bind it now : no other pix shader will replace it...
    g_pd3dDeviceContext->PSSetShader(g_pPixelShader, NULL, 0);
  }
  // Create the constant buffer
  {
    D3D11_BUFFER_DESC cbDesc;
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags =
        D3D11_BIND_CONSTANT_BUFFER;  // D3D11_BIND_SHADER_RESOURCE;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    cbDesc.MiscFlags = 0;
    cbDesc.ByteWidth = 16 * ((sizeof(ConstantBuffer) + 16) / 16);
    // cbDesc.StructureByteStride = 0;
    hr = g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pConstantBuffer);
    AssertOrQuit(SUCCEEDED(hr));
    // Assign the buffer now : nothing in the code will interfere with this
    // (very simple sample)
    g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
    g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);
  }
  // SamplerState
  {
    D3D11_SAMPLER_DESC sDesc;
    sDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sDesc.MinLOD = 0;
    sDesc.MaxLOD = 8;
    sDesc.MipLODBias = 0;
    sDesc.MaxAnisotropy = 1;
    hr = g_pd3dDevice->CreateSamplerState(&sDesc, &g_pSamplerState);
    AssertOrQuit(SUCCEEDED(hr));
    g_pd3dDeviceContext->PSSetSamplers(0, 1, &g_pSamplerState);
  }
#endif
  // Setup  no Input Layout
  g_pd3dDeviceContext->IASetInputLayout(0);
  g_pd3dDeviceContext->IASetPrimitiveTopology(
      D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

  D3D11_RASTERIZER_DESC rasterizerState;
  rasterizerState.FillMode = D3D11_FILL_SOLID;
  rasterizerState.CullMode = D3D11_CULL_FRONT;
  rasterizerState.FrontCounterClockwise = false;
  rasterizerState.DepthBias = false;
  rasterizerState.DepthBiasClamp = 0;
  rasterizerState.SlopeScaledDepthBias = 0;
  rasterizerState.DepthClipEnable = false;
  rasterizerState.ScissorEnable = false;
  rasterizerState.MultisampleEnable = false;
  rasterizerState.AntialiasedLineEnable = false;
  g_pd3dDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
  g_pd3dDeviceContext->RSSetState(g_pRasterState);

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: InitTexture3D()
// Desc: Initializes Direct3D Textures (allocation and initialization)
//-----------------------------------------------------------------------------
HRESULT InitTexture3D(const int width, const int height, const int depth, Texture3D& g_texture_3d) {
  //
  // create the D3D resources we'll be using
  //
  // 3D texture
  {
    g_texture_3d.width = width;
    g_texture_3d.height = height;
    g_texture_3d.depth = depth;

    D3D11_TEXTURE3D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE3D_DESC));
    desc.Width = g_texture_3d.width;
    desc.Height = g_texture_3d.height;
    desc.Depth = g_texture_3d.depth;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16_UNORM;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    if (FAILED(g_pd3dDevice->CreateTexture3D(&desc, NULL,
                                             &g_texture_3d.pTexture))) {
      return E_FAIL;
    }
  }

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void CleanupTexture(Texture3D& g_texture_3d) {
  //// unregister the Cuda resources
  //cudaGraphicsUnregisterResource(g_texture_3d.cudaResource);
  //getLastCudaError("cudaGraphicsUnregisterResource (g_texture_3d) failed");

  if (g_texture_3d.cuda_mip_mapped_array != 0) {
    cudaFreeMipmappedArray(g_texture_3d.cuda_mip_mapped_array);
    g_texture_3d.cuda_mip_mapped_array = 0;
  }
  if (g_texture_3d.cuda_external_memory != nullptr) {
    cudaDestroyExternalMemory(g_texture_3d.cuda_external_memory);
    g_texture_3d.cuda_external_memory = nullptr;
  }

  //
  // clean up Direct3D
  //
  {
    g_texture_3d.pTexture->Release();
  }
}

void CleanupOthers(){
  //
  // clean up Direct3D
  //
  {
#ifdef USEEFFECT

    if (g_pSimpleEffect != NULL) {
      g_pSimpleEffect->Release();
    }

#else

    if (g_pVertexShader) {
      g_pVertexShader->Release();
    }

    if (g_pPixelShader) {
      g_pPixelShader->Release();
    }

    if (g_pConstantBuffer) {
      g_pConstantBuffer->Release();
    }

    if (g_pSamplerState) {
      g_pSamplerState->Release();
    }

#endif

    if (g_pSwapChainRTV != NULL) {
      g_pSwapChainRTV->Release();
    }

    if (g_pSwapChain != NULL) {
      g_pSwapChain->Release();
    }

    if (g_pd3dDevice != NULL) {
      g_pd3dDevice->Release();
    }
  }
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam,
                              LPARAM lParam) {
  switch (msg) {
    case WM_KEYDOWN:
      if (wParam == VK_ESCAPE) {
        g_bDone = true;
        //Cleanup();
        PostQuitMessage(0);
        return 0;
      }

      break;

    case WM_DESTROY:
      g_bDone = true;
      //Cleanup();
      PostQuitMessage(0);
      return 0;

    case WM_PAINT:
      ValidateRect(hWnd, NULL);
      return 0;
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
