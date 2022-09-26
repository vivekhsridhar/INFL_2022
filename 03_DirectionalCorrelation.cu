//
//  CrossCorrelation.cu
//  CrossCorrelation
//
//  Created by Vivek Sridhar on 29/06/17.
//  Copyright © 2017 Vivek Sridhar. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

template <typename T>
std::string to_string(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

long factorial(long val)
{
std::cout << val << "\n";
    long result = 1;
    for (long i = 1; i <= val; ++i)
    {
        result *= i;
    }

    return result;
}

long combination(long n, long r)
{
    return (factorial(n)) / ((factorial(n - r)) * factorial(r));
}

__global__ void kernel(float *x1, float *y1, float *x2, float *y2, float *res, int tau, int na_frames, long nElements)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (tau < 0)
    {
        if (index >= -tau+na_frames)
        {
            res[index] = x1[index] * x2[index + tau] + y1[index] * y2[index + tau];
        }
	   else res[index] = 0.0;
    }
    else
    {
        if (index < nElements - tau)
        {
            res[index] = x1[index] * x2[index + tau] + y1[index] * y2[index + tau];
        }
	   else res[index] = 0.0;
    }
}

// total measurement points in the time series is defined by nElements
#define M 1024      // number of threads per block
#define fps 10      // frames per second of input video (used to determine tau)
#define time 5      // time in seconds within which time delayed cross correlation is calculated (tau ranges from -time*fps to time*fps)
#define n_inds 10

int na_frames = 0;  // number of frames in the start with nas
int scale = 1;    // time window for analysis in seconds; varying this allows us to examine dynamics of leadership across varying timescales; setting scale larger than the entire time series or -1 gives aggregated statistics across the entire duration (otherwise, timescale of analysis is scale*fps)

//const int pairs = combination(n_inds, 2);
const bool aggregate = false;	// this boolean decides whether you output a dynamic time variable leadership network or a static time aggregated network; scale is set to -1 if aggregate is true

std::ofstream outputFile1;

int main () 
{
    DIR *dir;
    FILE *pFile_x1; FILE *pFile_y1; FILE *pFile_x2; FILE *pFile_y2;
    long lSize;
    long nElements;
    struct dirent *file;
    
    float *d_x1, *d_y1, *d_x2, *d_y2, *d_res;
    float *x1, *y1, *x2, *y2, *res;

    size_t result_x1, result_y1, result_x2, result_y2;

    if (aggregate) scale = -1;
    
    std::vector<std::string> files;
    std::string directory = "/home/user/Documents/Vivek/cuda/DirectionalCorrelation/Data/Input/pigeons/10_birds/ffA3/cross_correlation/";
    dir = opendir(directory.c_str());

    int idx = 0;
    while ((file = readdir(dir)) != NULL)
    {
        if (file->d_name[0] == 'd')
        {
            files.push_back(file->d_name);
            ++idx;
        }
    }
    std::sort(files.begin(), files.begin()+2*n_inds);
    closedir(dir);

    // Open output file
    std::string filename_cc;
    if (scale != -1) filename_cc = "cross_correlation_01.csv";
    else filename_cc = "avgd_cross_correlation.csv";
    outputFile1.open(filename_cc.c_str());

    // Output file headers
    if (aggregate || scale == -1) outputFile1 << "id1"<< ", " << "id2"  << ", " << "tau" << ", " << "cc" << "\n";
    else outputFile1 << "time" << ", " << "id1" << ", " << "id2" << ", " << "tau" << ", " << "cc" << "\n";

    //files = {"dir_x00", "dir_x01", "dir_y00", "dir_y01"}

    for (int a = 0; a < n_inds; ++a)
    {
		for (int b = 0; b < n_inds; ++b)
		{
			if (b != a)
			{
				pFile_x1 = fopen ((directory + files[a]).c_str(), "rb");
	        	pFile_y1 = fopen ((directory + files[a+n_inds]).c_str(), "rb");
	        	pFile_x2 = fopen ((directory + files[b]).c_str(), "rb");
	        	pFile_y2 = fopen ((directory + files[b+n_inds]).c_str(), "rb");
	        	if (pFile_x1==NULL || pFile_y1==NULL || pFile_x2==NULL || pFile_y2==NULL) { fputs ("File error",stderr); exit (1); }
	        
        		// obtain file size
        		fseek (pFile_x1 , 0 , SEEK_END);
        		lSize = ftell (pFile_x1);
        		rewind (pFile_x1);
        
       			nElements = lSize / sizeof(float);

        		// allocate memory to contain the whole file
        		// device memory
        		cudaMalloc((void **) &d_x1, lSize);
        		cudaMalloc((void **) &d_y1, lSize);
        		cudaMalloc((void **) &d_x2, lSize);
        		cudaMalloc((void **) &d_y2, lSize);
        		cudaMalloc((void **) &d_res, lSize);

        		// host memory
        		x1 = (float*) malloc(lSize);
        		y1 = (float*) malloc(lSize);
        		x2 = (float*) malloc(lSize);
        		y2 = (float*) malloc(lSize);
        		res = (float*) malloc(lSize);
        		if (x1 == NULL || y1==NULL || x2==NULL || y2==NULL || res==NULL) { fputs ("Memory error",stderr); exit (2); }

        		// copy the file into the respective float pointers
        		result_x1 = fread (x1, sizeof(float), nElements, pFile_x1);
        		result_y1 = fread (y1, sizeof(float), nElements, pFile_y1);
        		result_x2 = fread (x2, sizeof(float), nElements, pFile_x2);
        		result_y2 = fread (y2, sizeof(float), nElements, pFile_y2);
        		if (result_x1 != nElements || result_y1 != nElements || result_x2 != nElements || result_y2 != nElements) { fputs ("Reading error",stderr); exit (3); }
        
        		// the whole files are now loaded in the memory x1, y1, x2 and y2 respectively
        
        		cudaMemcpy(d_x1, x1, lSize, cudaMemcpyHostToDevice);
        		cudaMemcpy(d_y1, y1, lSize, cudaMemcpyHostToDevice);
       			cudaMemcpy(d_x2, x2, lSize, cudaMemcpyHostToDevice);
        		cudaMemcpy(d_y2, y2, lSize, cudaMemcpyHostToDevice);
			
				if (scale*fps > nElements) scale = -1;

                int tau_max[nElements - scale*fps];
                float res_tmp[nElements - scale*fps];
                float res_max[nElements - scale*fps];
                std::fill_n(tau_max, nElements - scale*fps, 0);
                std::fill_n(res_tmp, nElements - scale*fps, 0.0);
                std::fill_n(res_max, nElements - scale*fps, -1.0);

				for (int tau = -time*fps; tau <= time*fps; ++tau) 
                {
                    kernel<<<(nElements + M - 1) / M, M>>>(d_x1, d_y1, d_x2, d_y2, d_res, tau, na_frames, nElements);
                    cudaMemcpy(res, d_res, lSize, cudaMemcpyDeviceToHost);

                    if (scale == -1)
                    {
                        float res_now = -1.0f;
                        for (int i = na_frames; i < nElements; ++i) 
			{
				if (res[i] != res[i]) std::cout << x1[i] << " " << y1[i] << " " << i << " " << tau << "\n";       // if nans
				res_now += res[i];
			}

                        outputFile1 << (to_string(files[a][5])).c_str() << (to_string(files[a][6])).c_str() << (to_string(files[a][7])).c_str()  << ", " << (to_string(files[b][5])).c_str() << (to_string(files[b][6])).c_str() << (to_string(files[b][7])).c_str() << ", " << tau << ", " << res_now / nElements << "\n";
                    }
                    else
                    {
			std::fill_n(res_tmp, nElements - scale*fps, 0.0);
                        for (int i = na_frames; i < nElements - scale*fps; ++i)
                        {
                            for (int j = i; j < i + scale*fps; ++j)
                            {
                                res_tmp[i] += res[j];
                                if (j == i + scale*fps - 1 && res_max[i] < res_tmp[i]) { res_max[i] = res_tmp[i]; tau_max[i] = tau; }
                            }
                        }
                    }
                }

                if (scale != -1) 
                {
                    for (int t = 0; t < nElements - scale*fps; ++t)
                    {
                        outputFile1 << t  + scale*fps/2 << ", " << (to_string(files[a][5])).c_str() << (to_string(files[a][6])).c_str() << (to_string(files[a][7])).c_str() << ", " << (to_string(files[b][5])).c_str() << (to_string(files[b][6])).c_str() << (to_string(files[b][7])).c_str() << ", " << tau_max[t] << ", " << res_max[t] / (scale*fps) << "\n";
                    }
                }

		fclose(pFile_x1);
		fclose(pFile_x2);
		fclose(pFile_y1);
		fclose(pFile_y2);

		cudaFree(d_x1); cudaFree(d_y1); cudaFree(d_x2); cudaFree(d_y2); cudaFree(d_res);
		free(x1); free(y1); free(x2); free(y2);
			}
		}
    }
    
    // terminate
    fclose(pFile_x1); fclose(pFile_y1); fclose(pFile_x2); fclose(pFile_y2);
    
    return 0;
}
