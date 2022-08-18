#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    // TODO Fill this in
    return im.data[im.w*im.h*c + im.w*y + x];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // TODO Fill this in
    im.data[im.w*im.h*c + im.w*y + x] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    // TODO Fill this in
    for(int c = 0; c < im.c; c++)
    {
        for(int y = 0; y < im.h; y++)
        {
            for(int x = 0; x < im.w; x++)
            {
                copy.data[im.w*im.h*c + im.w*y + x] = im.data[im.w*im.h*c + im.w*y + x];
            }
        }
    }
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    // TODO Fill this in
    for(int y = 0; y < im.h; y++)
    {
        for(int x = 0; x < im.w; x++)
        {
            float R = get_pixel(im, x, y, 0);
            float G = get_pixel(im, x, y, 1);
            float B = get_pixel(im, x, y, 2);
            float Y = 0.299 * R + 0.587 * G + 0.114 * B;
            set_pixel(gray, x, y, 0, Y);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    for(int y = 0; y < im.h; y++)
    {
        for(int x = 0; x < im.w; x++)
        {
            set_pixel(im, x, y, c, get_pixel(im, x, y, c)+v);
        }
    }
}

void clamp_image(image im)
{
    // TODO Fill this in
    for(int c = 0; c < im.c; c++)
    {
        for(int y = 0; y < im.h; y++)
        {
            for(int x = 0; x < im.w; x++)
            {
                if(im.data[im.w*im.h*c + im.w*y + x] < 0.0)
                {
                    im.data[im.w*im.h*c + im.w*y + x] = 0.0;
                }
                else if(im.data[im.w*im.h*c + im.w*y + x] > 1.0)
                {
                    im.data[im.w*im.h*c + im.w*y + x] = 1.0;
                }
            }
        }
    }

}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    for(int y = 0; y < im.h; y++)
    {
        for(int x = 0; x < im.w; x++)
        {
            float R = get_pixel(im, x, y, 0);
            float G = get_pixel(im, x, y, 1);
            float B = get_pixel(im, x, y, 2);

            float V = three_way_max(R, G, B);
            float m = three_way_min(R, G, B);
            float C = V - m;
            float S = C / V;

            float H_prime = 0.0;
            float H = 0.0;
            
            if(C == 0)
            {
                H_prime = 0.0;
            }
            else if(V == R)
            {
                H_prime = (G - B) / C;
            }
            else if(V == G)
            {
                H_prime = (B - R) / C + 2.0;
            }
            else if(V == B)
            {
                H_prime = (R - G) / C + 4.0;
            }
            
            if(H_prime < 0.0)
            {
                H = H_prime / 6.0 + 1.0;
            }
            else
            {
                H = H_prime / 6.0;
            }

            set_pixel(im, x, y, 0, H);
            set_pixel(im, x, y, 1, S);
            set_pixel(im, x, y, 2, V);
        }
    }
}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    for(int y = 0; y < im.h; y++)
    {
        for(int x = 0; x < im.w; x++)
        {
            float H = get_pixel(im, x, y, 0);
            float S = get_pixel(im, x, y, 1);
            float V = get_pixel(im, x, y, 2);
            float m = V - S*V;

            float R;
            float G;
            float B;

            if( (5.0/6.0) <= H && H < 1.0)
            {
                R = V;
                G = m;
                B = G - 6.0*(H - 1.0)*(V - m);
            }
            else if( 0.0 <= H && H < (1.0/6.0))
            {
                R = V;
                B = m;
                G = B + 6.0*H*(V - m);
            }
            else if( (1.0/6.0) <= H && H < (2.0/6.0))
            {
                G = V;
                B = m;
                R = B - (V - m)*(6.0*H - 2.0);
            }
            else if( (2.0/6.0) <= H && H < (3.0/6.0))
            {
                G = V;
                R = m;
                B = R + (V - m)*(6.0*H - 2.0);
            }
            else if( (3.0/6.0) <= H && H < (4.0/6.0))
            {
                B = V;
                R = m;
                G = R - (V - m)*(6.0*H - 4.0);
            }
            else if( (4.0/6.0) <= H && H < (5.0/6.0))
            {
                B = V;
                G = m;
                R = G + (V - m)*(6.0*H - 4.0);
            }

            set_pixel(im, x, y, 0, R);
            set_pixel(im, x, y, 1, G);
            set_pixel(im, x, y, 2, B);
        }
    }
}
