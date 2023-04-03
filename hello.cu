#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>

// Для использования просто скомпилируйте программу и выставите в main() входные поля SampleImageF1name[] и SampleImageF2name путь к изображениям одного размера
// Например 
//     char SampleImageF1name[] = "nature.bmp";
//     char SampleImageF2name[] = "dandelion.bmp";
// Или
//     char SampleImageF1name[] = "gradient.bmp";
//     char SampleImageF2name[] = "teapot512.bmp";
// Выходом программы являются два идентичных файла (один сформирован через CUDA, второй через обычные вычисления на CPU)
// А также - главное - разница во времени в выполнении операций на CPU и GPU

using namespace std;

#define BLOCK_SIZE 8

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma pack(push)
#endif
#pragma pack(1)

typedef char int8;
typedef short int16;
typedef int int32;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef unsigned char byte;

typedef struct {
    uint16 _bm_signature;    //!< File signature, must be "BM"
    uint32 _bm_file_size;    //!< File size
    uint32 _bm_reserved;     //!< Reserved, must be zero
    uint32 _bm_bitmap_data;  //!< Bitmap data
} BMPFileHeader;

typedef struct {
    uint32 _bm_info_header_size;      //!< Info header size, must be 40
    uint32 _bm_image_width;           //!< Image width
    uint32 _bm_image_height;          //!< Image height
    uint16 _bm_num_of_planes;         //!< Amount of image planes, must be 1
    uint16 _bm_color_depth;           //!< Color depth
    uint32 _bm_compressed;            //!< Image compression, must be none
    uint32 _bm_bitmap_size;           //!< Size of bitmap data
    uint32 _bm_hor_resolution;        //!< Horizontal resolution, assumed to be 0
    uint32 _bm_ver_resolution;        //!< Vertical resolution, assumed to be 0
    uint32 _bm_num_colors_used;       //!< Number of colors used, assumed to be 0
    uint32 _bm_num_important_colors;  //!< Number of important colors, assumed to be 0
} BMPInfoHeader;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma pack(pop)
#else
#pragma pack()
#endif

typedef struct {
    int width;   //!< ROI width
    int height;  //!< ROI height
} ROI;

int clamp_0_255(int x);
byte *MallocPlaneByte(int width, int height, int *pStepBytes);
void FreePlane(void *ptr);
int PreLoadBmp(char *FileName, int *Width, int *Height);
void LoadBmp(char *FileName, int Stride, ROI ImSize, byte *Img);
void DumpBmp(char *FileName, byte *Img, int Stride, ROI ImSize);

__global__ void overlayByteArrays(byte *a, byte *b, byte *c, ROI Size, int threads_count = 0){
    int idx = threadIdx.x;
    for (int i = idx; i < Size.height * Size.width * 3; i += threads_count){
        c[i] = (b[i] + a[i]) / 2;
    }
}

double overlay_images(byte *ImgSrc1, byte *ImgSrc2, byte *ImgDst, int Stride, ROI Size) {
    // create and start timer
    clock_t t1, t2;
    double time_diff;
    t1 = clock();

    // perform overlay
    for (int i = 0; i < Size.height; i++) {
        for (int j = 0; j < Size.width; j++) {
            ImgDst[i * Stride + j * 3] = (ImgSrc1[i * Stride + j * 3] + ImgSrc2[i * Stride + j * 3]) / 2;
            ImgDst[i * Stride + j * 3 + 1] = (ImgSrc1[i * Stride + j * 3 + 1] + ImgSrc2[i * Stride + j * 3 + 1]) / 2;
            ImgDst[i * Stride + j * 3 + 2] = (ImgSrc1[i * Stride + j * 3 + 2] + ImgSrc2[i * Stride + j * 3 + 2]) / 2;
        }
    } 

    // stop and destroy timer
    t2 = clock();
    time_diff = ((double)(t2 - t1)) / CLOCKS_PER_SEC;

    // return time taken by the operation
    return time_diff;
}

double overlay_images_CUDA(byte *ImgSrc1, byte *ImgSrc2, byte *ImgDst, int Stride, ROI Size) {
    byte *da, *db, *dc;
    int size = (((int) ceil(Size.width / 16.0f)) * 16 * 3) * Size.height;

    //move to GPU
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);

    cudaMemcpy(da, ImgSrc1, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(db, ImgSrc2, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    
    // create and start timer
    clock_t t1, t2;
    double time_diff;
    t1 = clock();

    // perform overlay
    int threads_count = 1 << 10;
    overlayByteArrays<<<1, threads_count>>>(da, db, dc, Size, threads_count);

    // stop and destroy timer
    t2 = clock();
    time_diff = ((double)(t2 - t1)) / CLOCKS_PER_SEC;

    //get result
    cudaMemcpy(ImgDst, dc, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    //Free buffers
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    // return time taken by the operation
    return time_diff;
}

int main(){
    char SampleImageF1name[] = "nature.bmp";
    char SampleImageF2name[] = "dandelion.bmp";
    char SampleImageFnameRes[] = "res.bmp";
    char SampleImageFnameResCuda[] = "res_CUDA.bmp";
    char *pSampleImageF1path = SampleImageF1name;
    char *pSampleImageF2path = SampleImageF2name;

    // preload image (acquire dimensions)
    int ImgWidth, ImgHeight;
    int ImgWidth1, ImgHeight1;
    ROI ImgSize;
    int res = PreLoadBmp(pSampleImageF1path, &ImgWidth, &ImgHeight);
    int res2 = PreLoadBmp(pSampleImageF2path, &ImgWidth1, &ImgHeight1);
    ImgSize.width = ImgWidth;
    ImgSize.height = ImgHeight;

    // CONSOLE INFORMATION: saying hello to user
    printf("Loading test images: %s and %s... ", SampleImageF1name, SampleImageF2name);
    if (res) {
        printf("\nError %d: Image file not found or invalid!\n", res);
        exit(EXIT_FAILURE);
        return 1;
    }

    if (ImgHeight != ImgHeight1 || ImgWidth != ImgWidth1){
        printf("\nError: Input image dimensions must be identical!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    // check image dimensions are multiples of BLOCK_SIZE
    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0) {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    printf("[%d x %d]... \n", ImgWidth, ImgHeight);

    // allocate image buffers
    int ImgStride;
    byte *ImgSrc1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgSrc2 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDst = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstCUDA = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);

    // load images
    LoadBmp(pSampleImageF1path, ImgStride, ImgSize, ImgSrc1);
    LoadBmp(pSampleImageF2path, ImgStride, ImgSize, ImgSrc2);

    //Running CPU version
    printf("Success\nRunning CPU version... \n");
    double TimeCPU = overlay_images(ImgSrc1, ImgSrc2, ImgDst, ImgStride, ImgSize);

    //Running GPU version
    printf("Success\nRunning GPU version... \n");
    double TimeGPU = overlay_images_CUDA(ImgSrc1, ImgSrc2, ImgDstCUDA, ImgStride, ImgSize);

    printf("Success\nDumping result to %s... \n", SampleImageFnameRes);
    DumpBmp(SampleImageFnameRes, ImgDst, ImgStride, ImgSize);

    printf("Success\nDumping result to %s... \n", SampleImageFnameResCuda);
    DumpBmp(SampleImageFnameResCuda, ImgDstCUDA, ImgStride, ImgSize);

    //Finalize
    FreePlane(ImgSrc1);
    FreePlane(ImgSrc2);
    FreePlane(ImgDst);
    FreePlane(ImgDstCUDA);

    printf("Processing time (CPU)    : %f ms \n", TimeCPU);
    printf("Processing time (GPU)    : %f ms \n", TimeGPU);

    return 0;
}

int clamp_0_255(int x) {
    return (x < 0) ? 0 : ((x > 255) ? 255 : x); 
}

byte *MallocPlaneByte(int width, int height, int *pStepBytes) {
    byte *ptr;
    *pStepBytes = ((int) ceil(width / 16.0f)) * 16 * 3;
    ptr = (byte *) malloc(*pStepBytes * height);
    return ptr;
}

void FreePlane(void *ptr) {
  if (ptr) {
    free(ptr);
  }
}

int PreLoadBmp(char *FileName, int *Width, int *Height) {
    BMPFileHeader FileHeader;
    BMPInfoHeader InfoHeader;
    FILE *fh;

    if (!(fh = fopen(FileName, "rb"))) {
        return 1;  // invalid filename
    }

    fread(&FileHeader, sizeof(BMPFileHeader), 1, fh);

    if (FileHeader._bm_signature != 0x4D42) {
        return 2;  // invalid file format
    }

    fread(&InfoHeader, sizeof(BMPInfoHeader), 1, fh);

    if (InfoHeader._bm_color_depth != 24) {
        printf("depth is %d\n", InfoHeader._bm_color_depth);
        return 3;  // invalid color depth
    }

    if (InfoHeader._bm_compressed) {
        printf("compression is %d\n", InfoHeader._bm_compressed);
        return 4;  // invalid compression property
    }

    *Width = InfoHeader._bm_image_width;
    *Height = InfoHeader._bm_image_height;

    fclose(fh);

    return 0;
}

void LoadBmp(char *FileName, int Stride, ROI ImSize, byte *Img) {
    BMPFileHeader FileHeader;
    BMPInfoHeader InfoHeader;
    FILE *fh;
    fh = fopen(FileName, "rb");

    fread(&FileHeader, sizeof(BMPFileHeader), 1, fh);
    fread(&InfoHeader, sizeof(BMPInfoHeader), 1, fh);

    for (int i = ImSize.height - 1; i >= 0; i--) {
        for (int j = 0; j < ImSize.width; j++) {
            int r = 0, g = 0, b = 0;
            fread(&b, 1, 1, fh);
            fread(&g, 1, 1, fh);
            fread(&r, 1, 1, fh);
            Img[i * Stride + j * 3] = b;
            Img[i * Stride + j * 3 + 1] = g;
            Img[i * Stride + j * 3 + 2] = r;
        }
    }

    fclose(fh);
    return;
}

void DumpBmp(char *FileName, byte *Img, int Stride, ROI ImSize) {
    FILE *fp = NULL;
    fp = fopen(FileName, "wb");

    if (fp == NULL) {
        return;
    }

    BMPFileHeader FileHeader;
    BMPInfoHeader InfoHeader;

    // init headers
    FileHeader._bm_signature = 0x4D42;
    FileHeader._bm_file_size = 54 + 3 * ImSize.width * ImSize.height;
    FileHeader._bm_reserved = 0;
    FileHeader._bm_bitmap_data = 0x36;
    InfoHeader._bm_bitmap_size = 0;
    InfoHeader._bm_color_depth = 24;
    InfoHeader._bm_compressed = 0;
    InfoHeader._bm_hor_resolution = 0;
    InfoHeader._bm_image_height = ImSize.height;
    InfoHeader._bm_image_width = ImSize.width;
    InfoHeader._bm_info_header_size = 40;
    InfoHeader._bm_num_colors_used = 0;
    InfoHeader._bm_num_important_colors = 0;
    InfoHeader._bm_num_of_planes = 1;
    InfoHeader._bm_ver_resolution = 0;

    fwrite(&FileHeader, sizeof(BMPFileHeader), 1, fp);
    fwrite(&InfoHeader, sizeof(BMPInfoHeader), 1, fp);

    for (int i = ImSize.height - 1; i >= 0; i--) {
        for (int j = 0; j < ImSize.width; j++) {
        fwrite(&(Img[i * Stride + j * 3]), 1, 1, fp);
        fwrite(&(Img[i * Stride + j * 3 + 1]), 1, 1, fp);
        fwrite(&(Img[i * Stride + j * 3 + 2]), 1, 1, fp);
        }
    }

    fclose(fp);
}
