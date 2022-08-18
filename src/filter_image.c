#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
    // TODO
    float sum = 0.0;
    // sum all the elements of image tensor
    for(int c_idx = 0; c_idx < im.c; c_idx++)
    {
        for(int h_idx = 0; h_idx < im.h; h_idx++)
        {
            for(int w_idx = 0; w_idx < im.w; w_idx++)
            {
                sum = sum + im.data[im.w * im.h * c_idx + im.w * h_idx + w_idx];
            }
        }
    }
    
    if(sum != 0.0)
    {
        for(int c_idx = 0; c_idx < im.c; c_idx++)
        {
            for(int h_idx = 0; h_idx < im.h; h_idx++)
            {
                for(int w_idx = 0; w_idx < im.w; w_idx++)
                {
                    im.data[im.w * im.h * c_idx + im.w * h_idx + w_idx]
                    = im.data[im.w * im.h * c_idx + im.w * h_idx + w_idx] * (1.0/sum);
                }
            }
        }
    }

}

image make_box_filter(int w)
{
    // TODO
    // make a box filter with dimension like [w x w x 1]
    image box_filter = make_image(w,w,1);
    float sum = (float) w * w;

    for(int c_idx = 0; c_idx < box_filter.c; c_idx++)
    {
        for(int h_idx = 0; h_idx < box_filter.h; h_idx++)
        {
            for(int w_idx = 0; w_idx < box_filter.w; w_idx++)
            {
                box_filter.data[box_filter.w * box_filter.h * c_idx + box_filter.w * h_idx + w_idx]
                 = 1.0/sum;
            }
        }
    }

    return box_filter;
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO

    assert(filter.c == im.c || filter.c == 1);
    
    int padding = (int)((filter.w-1.0)/2.0);// float padding = ((filter.h-1.0)/2.0); 으로 해도 됨

    // old_im : 원래 이미지인 im 에 zero padding 한 것
    image old_im = make_image(im.w + 2.0*padding, im.h + 2.0*padding, im.c);
    // new_im : filtering 을 거치고 나서 얻는 것
    image new_im;

    //old_im 모든 원소를 0으로 채우기
    for(int c = 0; c < old_im.c; c++)
    {
        for(int h = 0; h < old_im.h; h++)
        {
            for(int w = 0; w < old_im.w; w++)
            {
                old_im.data[old_im.w * old_im.h * c + old_im.w * h + w] = 0.0;
            }
        }
    }
    //old_im 에서 zero padding 영역 제외하고, 그 안쪽영역을 원래 im 의 원소들로 채우기 
    for(int c = 0; c < old_im.c; c++)
    {
        for(int h = padding; h < old_im.h - padding; h++)
        {
            for(int w = padding; w < old_im.w - padding; w++)
            {
                old_im.data[old_im.w * old_im.h * c + old_im.w * h + w]
                 = im.data[im.w * im.h * c + im.w * (h-padding) + (w-padding)];
            }
        }
    }


    if(im.c == filter.c)
    {
        if(preserve == 1) // case 1 //
        {
            // case 1 인 경우의 new_im dim 정의
            new_im = make_image(im.w, im.h, im.c);

            for(int c = 0; c < im.c; c++)
            {
                // im 위에서 filter 의 중심좌표를 sliding 하기 위한 looping
                for(int y = 0; y < im.h; y++)
                {
                    for(int x = 0; x < im.w; x++)
                    {
                        // filter 내에서 곱셈연산 수행할 좌표를 looping
                        float sum = 0.0;
                        for(int y_in_filter = 0; y_in_filter < filter.h; y_in_filter++)
                        {
                            for(int x_in_filter = 0; x_in_filter < filter.w; x_in_filter++)
                            {
                                sum = sum
                                 + filter.data[filter.w * filter.h * c + filter.w * y_in_filter + x_in_filter]
                                 * old_im.data[old_im.w * old_im.h * c + old_im.w * (y+y_in_filter) + (x+x_in_filter)];
                            }
                        }
                        new_im.data[im.w * im.h + c + im.w * y + x] = sum;
                    }
                }
            }

        }
        else // case 2 //
        {
            // case 2 인 경우의 new_im dim 정의
            new_im = make_image(im.w, im.h, 1);

            for(int y = 0; y < im.h; y++)
            {
                for(int x = 0; x < im.w; x++)
                {
                    // filter 내에서 곱셈연산 수행할 좌표를 looping
                    float sum = 0.0;
                    // 여기에 for문 하나더 감싸야 할듯!
                    for(int c = 0; c < im.c; c++)
                    {
                        for(int y_in_filter = 0; y_in_filter < filter.h; y_in_filter++)
                        {
                            for(int x_in_filter = 0; x_in_filter < filter.w; x_in_filter++)
                            {
                                sum = sum
                                    + filter.data[filter.w * filter.h * c + filter.w * y_in_filter + x_in_filter]
                                    * old_im.data[old_im.w * old_im.h * c + old_im.w * (y+y_in_filter) + (x+x_in_filter)];
                            }
                        }
                    }
                    new_im.data[im.w * im.h + 0 + im.w * y + x] = sum;
                }
            }
        }
    }
    else if(filter.c == 1)
    {
        if(preserve == 1) // case 3 //
        {
            // case 3 인 경우의 new_im dim 정의
            new_im = make_image(im.w, im.h, im.c);

            for(int c = 0; c < im.c; c++)
            {
                // im 위에서 filter 의 중심좌표를 sliding 하기 위한 looping
                for(int y = 0; y < im.h; y++)
                {
                    for(int x = 0; x < im.w; x++)
                    {
                        // filter 내에서 곱셈연산 수행할 좌표를 looping
                        float sum = 0.0;
                        for(int y_in_filter = 0; y_in_filter < filter.h; y_in_filter++)
                        {
                            for(int x_in_filter = 0; x_in_filter < filter.w; x_in_filter++)
                            {
                                sum = sum
                                 + filter.data[filter.w * filter.h * 0 + filter.w * y_in_filter + x_in_filter]
                                 * old_im.data[old_im.w * old_im.h * c + old_im.w * (y+y_in_filter) + (x+x_in_filter)];
                            }
                        }
                        new_im.data[new_im.w * new_im.h * c + new_im.w * y + x] = sum;
                    }
                }
            }
        }
        else // case 4 //
        {

            //tmp_im : channel이 1인 filter를 im의 각 channel 에서 여러번 사용하여 만든 중간 결과
            image tmp_im = make_image(im.w, im.h, im.c);
            for(int c = 0; c < im.c; c++)
            {
                // im 위에서 filter 의 중심좌표를 sliding 하기 위한 looping
                for(int y = 0; y < im.h; y++)
                {
                    for(int x = 0; x < im.w; x++)
                    {
                        // filter 내에서 곱셈연산 수행할 좌표를 looping
                        float sum = 0.0;
                        for(int y_in_filter = 0; y_in_filter < filter.h; y_in_filter++)
                        {
                            for(int x_in_filter = 0; x_in_filter < filter.w; x_in_filter++)
                            {
                                sum = sum
                                 + filter.data[filter.w * filter.h * 0 + filter.w * y_in_filter + x_in_filter]
                                 * old_im.data[old_im.w * old_im.h * c + old_im.w * (y+y_in_filter) + (x+x_in_filter)];
                            }
                        }
                        tmp_im.data[tmp_im.w * tmp_im.h * c + tmp_im.w * y + x] = sum;
                    }
                } 
            }

            // case 4 인 경우의 new_im dim 정의
            // 중간결과인 tmp_im 을 channel 방향으로 압축(sum between channels)하여 최종결과인 new_im 을 만든다
            new_im = make_image(im.w, im.h, 1);

            // tmp_im 을 channel 방향으로 압축한 결과를 new_im 에 저장
            for(int y = 0; y < tmp_im.h; y++)
            {
                for(int x = 0; x < tmp_im.w; x++)
                {
                    float sum = 0.0;
                    for(int c = 0; c < tmp_im.c; c++)
                    {
                        sum = sum + tmp_im.data[tmp_im.w * tmp_im.h * c + tmp_im.w * y + x];
                    }
                    new_im.data[new_im.w * new_im.h * 0 + new_im.w * y + x] = sum;
                }
            }

        }
    }
    return new_im;
}

image make_highpass_filter()
{
    // TODO
    image im = make_image(3,3,1);
    im.data[im.w * im.h * 0 + im.w * 0 + 0] = 0.0; // when (channel,row,column) = (0,0,0)
    im.data[im.w * im.h * 0 + im.w * 0 + 1] = -1.0; // when (channel,row,column) = (0,0,1)
    im.data[im.w * im.h * 0 + im.w * 0 + 2] = 0.0; // when (channel,row,column) = (0,0,2)
    im.data[im.w * im.h * 0 + im.w * 1 + 0] = -1.0; // when (channel,row,column) = (0,1,0)
    im.data[im.w * im.h * 0 + im.w * 1 + 1] = 4.0; // when (channel,row,column) = (0,1,1)
    im.data[im.w * im.h * 0 + im.w * 1 + 2] = -1.0; // when (channel,row,column) = (0,1,2)
    im.data[im.w * im.h * 0 + im.w * 2 + 0] = 0.0; // when (channel,row,column) = (0,2,0)
    im.data[im.w * im.h * 0 + im.w * 2 + 1] = -1.0; // when (channel,row,column) = (0,2,1)
    im.data[im.w * im.h * 0 + im.w * 2 + 2] = 0.0; // when (channel,row,column) = (0,2,2)

    return im;
}

image make_sharpen_filter()
{
    // TODO
    image im = make_image(3,3,1);
    im.data[im.w * im.h * 0 + im.w * 0 + 0] = 0.0; // when (channel,row,column) = (0,0,0)
    im.data[im.w * im.h * 0 + im.w * 0 + 1] = -1.0; // when (channel,row,column) = (0,0,1)
    im.data[im.w * im.h * 0 + im.w * 0 + 2] = 0.0; // when (channel,row,column) = (0,0,2)
    im.data[im.w * im.h * 0 + im.w * 1 + 0] = -1.0; // when (channel,row,column) = (0,1,0)
    im.data[im.w * im.h * 0 + im.w * 1 + 1] = 5.0; // when (channel,row,column) = (0,1,1)
    im.data[im.w * im.h * 0 + im.w * 1 + 2] = -1.0; // when (channel,row,column) = (0,1,2)
    im.data[im.w * im.h * 0 + im.w * 2 + 0] = 0.0; // when (channel,row,column) = (0,2,0)
    im.data[im.w * im.h * 0 + im.w * 2 + 1] = -1.0; // when (channel,row,column) = (0,2,1)
    im.data[im.w * im.h * 0 + im.w * 2 + 2] = 0.0; // when (channel,row,column) = (0,2,2)

    return im;
}

image make_emboss_filter()
{
    // TODO
    image im = make_image(3,3,1);
    im.data[im.w * im.h * 0 + im.w * 0 + 0] = -2.0; // when (channel,row,column) = (0,0,0)
    im.data[im.w * im.h * 0 + im.w * 0 + 1] = -1.0; // when (channel,row,column) = (0,0,1)
    im.data[im.w * im.h * 0 + im.w * 0 + 2] = 0.0; // when (channel,row,column) = (0,0,2)
    im.data[im.w * im.h * 0 + im.w * 1 + 0] = -1.0; // when (channel,row,column) = (0,1,0)
    im.data[im.w * im.h * 0 + im.w * 1 + 1] = 1.0; // when (channel,row,column) = (0,1,1)
    im.data[im.w * im.h * 0 + im.w * 1 + 2] = 1.0; // when (channel,row,column) = (0,1,2)
    im.data[im.w * im.h * 0 + im.w * 2 + 0] = 0.0; // when (channel,row,column) = (0,2,0)
    im.data[im.w * im.h * 0 + im.w * 2 + 1] = 1.0; // when (channel,row,column) = (0,2,1)
    im.data[im.w * im.h * 0 + im.w * 2 + 2] = 2.0; // when (channel,row,column) = (0,2,2)

    return im;
}

// Question 2.2.1: Which of these filters should we use preserve 
// when we run our convolution and which ones should we not? Why?
// Answer: highpass_filter - no preserve, sharpen_filter - preserve, emboss_filter - preserve

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: Clamp, cuz it needs to fix overflow and underflow of pixel values in the result images.

image make_gaussian_filter(float sigma)
{
    // TODO
    image new_im = make_image(2*sigma+1, 2*sigma+1, 1);

    float sum = 0.0;
    for(int y = 0; y < new_im.h; y++)
    {
        for(int x = 0; x < new_im.w; x++)
        {
            new_im.data[new_im.w * new_im.h * 0 + new_im.w * y + x]
             = (1.0/(TWOPI*sigma*sigma)) * exp(-1.0*( (((float)x-sigma)*((float)x-sigma) + ((float)y-sigma)*((float)y-sigma))/(2.0*sigma*sigma) ));
            sum = sum + (1.0/(TWOPI*sigma*sigma)) * exp(-1.0*( (((float)x-sigma)*((float)x-sigma) + ((float)y-sigma)*((float)y-sigma))/(2.0*sigma*sigma) ));
        }
    }

    for(int y = 0; y < new_im.h; y++)
    {
        for(int x = 0; x < new_im.w; x++)
        {
            new_im.data[new_im.w * new_im.h * 0 + new_im.w * y + x]
             = (1.0/sum) * new_im.data[new_im.w * new_im.h * 0 + new_im.w * y + x];
        }
    }

    return new_im;
}

image add_image(image a, image b)
{
    // TODO
    assert(a.h == b.h && a.w == b.w);

    image new_im;

    if(a.c != b.c)
    {
        if(a.c > b.c && b.c == 1)
        {
            new_im = make_image(a.w, a.h, a.c);

            for(int c = 0; c < new_im.c; c++)
            {
                for(int y = 0; y < new_im.h; y++)
                {
                    for(int x = 0; x < new_im.w; x++)
                    {
                        new_im.data[new_im.w * new_im.h * c + new_im.w * y + x]
                        = a.data[new_im.w * new_im.h * c + new_im.w * y + x] + b.data[new_im.w * new_im.h * 0 + new_im.w * y + x];
                    }
                }
            }

        }
        else if(b.c > a.c && a.c == 1) //when a.c < b.c
        {
            new_im = make_image(b.w, b.h, b.c);

            for(int c = 0; c < new_im.c; c++)
            {
                for(int y = 0; y < new_im.h; y++)
                {
                    for(int x = 0; x < new_im.w; x++)
                    {
                        new_im.data[new_im.w * new_im.h * c + new_im.w * y + x]
                        = a.data[new_im.w * new_im.h * 0 + new_im.w * y + x] + b.data[new_im.w * new_im.h * c + new_im.w * y + x];
                    }
                }
            }

        }
    }
    else if(a.c == b.c)
    {
        new_im = make_image(a.w, a.h, a.c);

        for(int c = 0; c < new_im.c; c++)
        {
            for(int y = 0; y < new_im.h; y++)
            {
                for(int x = 0; x < new_im.w; x++)
                {
                    new_im.data[new_im.w * new_im.h * c + new_im.w * y + x]
                    = a.data[new_im.w * new_im.h * c + new_im.w * y + x] + b.data[new_im.w * new_im.h * c + new_im.w * y + x];
                }
            }
        }
    }

    return new_im;
}

image sub_image(image a, image b)
{
    // TODO
    assert(a.h == b.h && a.w == b.w);

    image new_im;

    if(a.c != b.c)
    {
        if(a.c > b.c && b.c == 1)
        {
            new_im = make_image(a.w, a.h, a.c);

            for(int c = 0; c < new_im.c; c++)
            {
                for(int y = 0; y < new_im.h; y++)
                {
                    for(int x = 0; x < new_im.w; x++)
                    {
                        new_im.data[new_im.w * new_im.h * c + new_im.w * y + x]
                        = a.data[new_im.w * new_im.h * c + new_im.w * y + x] - b.data[new_im.w * new_im.h * 0 + new_im.w * y + x];
                    }
                }
            }

        }
        else if(b.c > a.c && a.c == 1) //when a.c < b.c
        {
            new_im = make_image(b.w, b.h, b.c);

            for(int c = 0; c < new_im.c; c++)
            {
                for(int y = 0; y < new_im.h; y++)
                {
                    for(int x = 0; x < new_im.w; x++)
                    {
                        new_im.data[new_im.w * new_im.h * c + new_im.w * y + x]
                        = a.data[new_im.w * new_im.h * 0 + new_im.w * y + x] - b.data[new_im.w * new_im.h * c + new_im.w * y + x];
                    }
                }
            }

        }
    }
    else if(a.c == b.c)
    {
        new_im = make_image(a.w, a.h, a.c);

        for(int c = 0; c < new_im.c; c++)
        {
            for(int y = 0; y < new_im.h; y++)
            {
                for(int x = 0; x < new_im.w; x++)
                {
                    new_im.data[new_im.w * new_im.h * c + new_im.w * y + x]
                    = a.data[new_im.w * new_im.h * c + new_im.w * y + x] - b.data[new_im.w * new_im.h * c + new_im.w * y + x];
                }
            }
        }
    }

    return new_im;
}

image make_gx_filter()
{
    // TODO
    image im = make_image(3,3,1);
    im.data[im.w * im.h * 0 + im.w * 0 + 0] = -1.0;
    im.data[im.w * im.h * 0 + im.w * 0 + 1] = 0.0;
    im.data[im.w * im.h * 0 + im.w * 0 + 2] = 1.0;
    im.data[im.w * im.h * 0 + im.w * 1 + 0] = -2.0;
    im.data[im.w * im.h * 0 + im.w * 1 + 1] = 0.0;
    im.data[im.w * im.h * 0 + im.w * 1 + 2] = 2.0;
    im.data[im.w * im.h * 0 + im.w * 2 + 0] = -1.0;
    im.data[im.w * im.h * 0 + im.w * 2 + 1] = 0.0;
    im.data[im.w * im.h * 0 + im.w * 2 + 2] = 1.0;

    return im;
}

image make_gy_filter()
{
    // TODO
    image im = make_image(3,3,1);
    im.data[im.w * im.h * 0 + im.w * 0 + 0] = -1.0;
    im.data[im.w * im.h * 0 + im.w * 0 + 1] = -2.0;
    im.data[im.w * im.h * 0 + im.w * 0 + 2] = -1.0;
    im.data[im.w * im.h * 0 + im.w * 1 + 0] = 0.0;
    im.data[im.w * im.h * 0 + im.w * 1 + 1] = 0.0;
    im.data[im.w * im.h * 0 + im.w * 1 + 2] = 0.0;
    im.data[im.w * im.h * 0 + im.w * 2 + 0] = 1.0;
    im.data[im.w * im.h * 0 + im.w * 2 + 1] = 2.0;
    im.data[im.w * im.h * 0 + im.w * 2 + 2] = 1.0;

    return im;
}

void feature_normalize(image im)
{
    // TODO
    float min = 10000.0;
    float max = -10000.0;
    for(int c = 0; c < im.c; c++)
    {
        for(int y = 0; y < im.h; y++)
        {
            for(int x = 0; x < im.w; x++)
            {
                if(im.data[im.w * im.h * c + im.w * y + x] < min)
                {
                    min = im.data[im.w * im.h * c + im.w * y + x];
                }
                if(im.data[im.w * im.h * c + im.w * y + x] > max)
                {
                    max = im.data[im.w * im.h * c + im.w * y + x];
                }
            }
        }
    }

    for(int c = 0; c < im.c; c++)
    {
        for(int y = 0; y < im.h; y++)
        {
            for(int x = 0; x < im.w; x++)
            {
                im.data[im.w * im.h * c + im.w * y + x]
                 = (im.data[im.w * im.h * c + im.w * y + x] - min) / (max - min);
            }
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    image* im_pointer = (image*) calloc(2, sizeof(image)); // "두개의 image 구조체를 담는 image 구조체 배열" 을 가리키는 image 구조체 포인터 im_pointer 선언 

    image magnitude = make_image(im.w, im.h, 1);
    image direction = make_image(im.w, im.h, 1);

    image gx_filter = make_gx_filter();
    image gy_filter = make_gy_filter();

    image gx_im = convolve_image(im, gx_filter, 0);
    image gy_im = convolve_image(im, gy_filter, 0);

    for(int y = 0; y < im.h; y++)
    {
        for(int x = 0; x < im.w; x++)
        {
            float gx = gx_im.data[gx_im.w * gx_im.h * 0 + gx_im.w * y + x];
            float gy = gy_im.data[gy_im.w * gy_im.h * 0 + gy_im.w * y + x];
            
            magnitude.data[magnitude.w * magnitude.h * 0 + magnitude.w * y + x]
             = sqrt(gx*gx + gy*gy);
            direction.data[direction.w * direction.h * 0 + direction.w * y + x]
             = atan2(gy, gx);
        }
    }

    im_pointer[0] = magnitude;
    im_pointer[1] = direction;


    return im_pointer;
}

image colorize_sobel(image im)
{
    // TODO
    image* sobel_img = sobel_image(im);

    image magnitude = sobel_img[0];
    image direction = sobel_img[1]; 

    // magnitude 와 direction 값을 반드시 normalize 해야 함. 
    // 그렇지않으면, 그 값이 매우 크게 튈 수 있고, 그로 인해 hsv 값으로 아주 큰 값이 저장될 수 있음.
    // hsv_to_rgb 함수는 hsv 값이 0.0 ~ 1.0 사이 범위를 가진다는 가정하에 코드가 작성되어 있으므로,
    // 매우 큰 값이 hsv 값으로 저장되면, hsv 에서 rgb 로 변환이 이루어지지 않을 수 있음.
    feature_normalize(magnitude); 
    feature_normalize(direction); 

    image hsv_matrix = make_image(im.w, im.h, im.c);

    
    // set HSV values
    for(int y = 0; y < im.h; y++)
    {
        for(int x = 0; x < im.w; x++)
        {
            float H = direction.data[im.w*im.h*0 + im.w*y + x];
            float S = magnitude.data[im.w*im.h*0 + im.w*y + x];
            float V = magnitude.data[im.w*im.h*0 + im.w*y + x];

            hsv_matrix.data[im.w*im.h*0 + im.w*y + x] = H;
            hsv_matrix.data[im.w*im.h*1 + im.w*y + x] = S;
            hsv_matrix.data[im.w*im.h*2 + im.w*y + x] = V;
        }
    }
    // convert HSV to RGB
    hsv_to_rgb(hsv_matrix);



    
    return hsv_matrix;
}
