#include <math.h>
#include "image.h"

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO Fill in
    // x : weird x value, y : weird y value
    x = x + 0.5;
    y = y + 0.5;

    int x_to_read = (int)x;
    int y_to_read = (int)y;

    return im.data[im.w*im.h*c + im.w*y_to_read + x_to_read];
}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix that first line)
    image new_im = make_image(w, h, im.c);

    for(int c_idx = 0; c_idx < new_im.c; c_idx++)
    {
        for(int h_idx = 0; h_idx < new_im.h; h_idx++)
        {
            for(int w_idx = 0; w_idx < new_im.w; w_idx++)
            {
                // w_idx and h_idx are looping over new image coord.
                float w_idx_in_old_coord = ((float)im.w/new_im.w)*w_idx - 0.5*(1.0 - ((float)im.w/new_im.w));
                float h_idx_in_old_coord = ((float)im.h/new_im.h)*h_idx - 0.5*(1.0 - ((float)im.h/new_im.h));

                new_im.data[new_im.w * new_im.h * c_idx + new_im.w * h_idx + w_idx]
                 = nn_interpolate(im, w_idx_in_old_coord, h_idx_in_old_coord, c_idx);
            }
        }
    }
    return new_im;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    // x : weird x value, y : weird y value
    int left_x = (int)x;
    int right_x = (int)x + 1;
    int top_y = (int)y;
    int bottom_y = (int)y + 1;

    float q1 = ( y-(top_y)) * im.data[im.w*im.h*c + im.w*bottom_y + left_x]
                + ( (bottom_y)-y ) * im.data[im.w*im.h*c + im.w*top_y + left_x];
    float q2 = ( y-(top_y) ) * im.data[im.w*im.h*c + im.w*bottom_y + right_x]
                + ( (bottom_y)-y ) * im.data[im.w*im.h*c + im.w*top_y + right_x];
    
    float q = ( x-(left_x) ) * q2 + ( (right_x)-x ) * q1;
    
    return q;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    image new_im = make_image(w, h, im.c);

    image tmp_im = make_image(im.w + 1, im.h + 1, im.c);
    //나중에 음수 index 에서도 bilinear interpolate 을 하기 위해, 아예 음수 index 가 안나오도록, 원래 im 보다 x축,y축 방향으로 한칸씩 더 큰 tmp_im 를 만든다.
    // 그리고 이렇게 새로만든 tmp_im 에서 bilinear interpolate 을 한다.
    for(int c_idx = 0; c_idx < tmp_im.c; c_idx++)
    {
        //tmp_im 의 (0,0),(0,1),(1,0) 는 im 의 (0,0) 픽셀값으로 padding 
        tmp_im.data[tmp_im.w * tmp_im.h * c_idx + tmp_im.w * 0 + 0] = im.data[im.w * im.h * c_idx + im.w * 0 + 0];
        tmp_im.data[tmp_im.w * tmp_im.h * c_idx + tmp_im.w * 0 + 1] = im.data[im.w * im.h * c_idx + im.w * 0 + 0];
        tmp_im.data[tmp_im.w * tmp_im.h * c_idx + tmp_im.w * 1 + 0] = im.data[im.w * im.h * c_idx + im.w * 0 + 0];

        //tmp_im 의 (...(>=2), 0 ) 는 im 의 (...(>=1), 0 ) 픽셀값으로 padding
        for(int w_idx = 2; w_idx < tmp_im.w;  w_idx++)
        {
            tmp_im.data[tmp_im.w * tmp_im.h * c_idx + tmp_im.w * 0 + w_idx] = im.data[im.w * im.h * c_idx + im.w * 0 + (w_idx-1)];
        }

        //tmp_im 의 (0, ...(>=2) ) 는 im 의 (0, ...(>=1) ) 픽셀값으로 padding
        for(int h_idx = 2; h_idx < tmp_im.h; h_idx++)
        {
            tmp_im.data[tmp_im.w * tmp_im.h * c_idx + tmp_im.w * h_idx + 0] = im.data[im.w * im.h * c_idx + im.w * (h_idx-1) + 0];
        }
    }
    // tmp_im 의 (...(>=1), ...(>=1)) 에  im 의 (...(>=0), ...(>=0)) 픽셀값을 할당
    for(int c_idx = 0; c_idx < tmp_im.c; c_idx++)
    {
        for(int h_idx = 1; h_idx < tmp_im.h; h_idx++)
        {
            for(int w_idx = 1; w_idx < tmp_im.w; w_idx++)
            {
                tmp_im.data[tmp_im.w * tmp_im.h * c_idx + tmp_im.w * h_idx + w_idx]
                 = im.data[im.w * im.h * c_idx + im.w * (h_idx-1) + (w_idx-1)];
            }
        }
    }


    for(int c_idx = 0; c_idx < new_im.c; c_idx++)
    {
        for(int h_idx = 0; h_idx < new_im.h; h_idx++)
        {
            for(int w_idx = 0; w_idx < new_im.w; w_idx++)
            {
                // w_idx and h_idx are looping over new image coord.
                float w_idx_in_old_coord = ((float)im.w/new_im.w)*w_idx - 0.5*(1.0 - ((float)im.w/new_im.w));
                float h_idx_in_old_coord = ((float)im.h/new_im.h)*h_idx - 0.5*(1.0 - ((float)im.h/new_im.h));

                new_im.data[new_im.w * new_im.h * c_idx + new_im.w * h_idx + w_idx]
                 = bilinear_interpolate(tmp_im, w_idx_in_old_coord + 1, h_idx_in_old_coord + 1, c_idx);
            }
        }
    }

    return new_im;
}

