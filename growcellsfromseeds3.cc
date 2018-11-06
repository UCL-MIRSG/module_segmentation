/*
 
 1. Given the seeds with a certain region
 2. Find the boundary of the cells by fully "growing them"
 
 */

#include "mex.h"
#include "utilah.h"

//pixel strucure
typedef struct
{
    int x;
    int y;
}pix;



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //plhs => left hand side/output; prhs => right hand side/ input
    matx *A, *B;
    matx *pixel_list;
    int mrows, ncols;
    int n,x,y,count, no_pixels1, no_pixels2, i;
    double level, celllevel, maxval;
    double *img;
    pix* pixlist;
    pix* pixlist2;
    pix* tmp;
    double cellID;
    
    
    // --- testing input and output values ---
    if (nrhs != 2) { mexErrMsgTxt(" usage: B=growcellsfromseeds2(CellImage with seeds at 255, celllevel)"); }
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("Image input must be a double real matrix");
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("Image input must be a double real matrix");
    
    // -- initialising arrays and stuff ---
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    A=mx_from_vector( mxGetPr(prhs[0]), mrows, ncols);
    
    img=mxGetPr(prhs[0]);
    
    //threshold that defines seed location (intensity range of seeds)
    celllevel = mxGetScalar( prhs[1]);
    
    
    plhs[0] = mxCreateDoubleMatrix( mrows,ncols, mxREAL);
    B=mx_from_vector( mxGetPr(plhs[0]), mrows, ncols);
    
    // estimate number of pixels per image for the pixellist
    // find the max value
    
    no_pixels1=0;
    maxval=0;
    
    for (n=0; n < mrows*ncols; n++)
    {
        if (img[n] > maxval) maxval=img[n];
    }
    
    // creating initial pixel list
    
    pixlist = (pix*)malloc(mrows*ncols*sizeof(pix));
    pixlist2 = (pix*)malloc(mrows*ncols*sizeof(pix));
    
    n=0;
    
    // initial populating pixlist and label matrix B
    // pixlist just contains the seed points
    for (y=0; y < A->rows; y++)
        for (x=0; x < A->cols; x++)
        {
            if (A->rptr[x][y] > celllevel)
            {
                pixlist[n].x=x;
                pixlist[n].y=y;
                B->rptr[x][y]=n+1; // assign new cell ID to output matrix
                n++; no_pixels1++;
            }
            else
                B->rptr[x][y]=0;
        }
    
    //printf("maxval: %d, n: %d\n", (int)maxval, no_pixels1);
    
    for (level=1; level <= maxval; level=level+0.05) //maxval
    {
        no_pixels2=0;
        
        // growing cells
        
        //2 pixel lists
        //alternative storing to avoid indefinitive growth
        //i.e. the label matrix is updated only at the end to not influence the current run
      
            for (n=0; n < no_pixels1; n++)
            {
                x=pixlist[n].x;  y=pixlist[n].y;
                
                //found new pixels and using
                pixlist2[no_pixels2].x=x;
                pixlist2[no_pixels2].y=y;
                
                no_pixels2++;
                cellID=B->rptr[x][y];
                
                // Are we touching another cell? If yes stop growing
                
                if(x < ncols-1 && B->rptr[x+1][y] > 0 && B->rptr[x+1][y] != cellID) { continue; }
                if(x > 0 && B->rptr[x-1][y] > 0 && B->rptr[x-1][y] != cellID) { continue; }
                if(y > 0 && B->rptr[x][y-1] > 0 && B->rptr[x][y-1] != cellID) { continue; }
                if(y < mrows-1 && B->rptr[x][y+1] > 0 && B->rptr[x][y+1] != cellID) { continue; }
                
                
                // trying to grow - are we having pixels :
                // - connecting unallocated pixels to cells [ B->rptr[x+1][y] < 1 ]
                // - below the current threshold level      [ A->rptr[x+1][y] < level ]
                
                
                if(x < ncols-1 && B->rptr[x+1][y] < 1 && A->rptr[x+1][y] < level) //grow by level again //todo avoid background explosion
                {
                    B->rptr[x+1][y] = cellID; //growth! new label: B->rptr[x][y]
                    pixlist2[no_pixels2].x=x+1;
                    pixlist2[no_pixels2].y=y;
                    no_pixels2++;   
                }
                
                
                if(x > 0 && B->rptr[x-1][y] < 1 && A->rptr[x-1][y] < level)
                {
                    B->rptr[x-1][y]=cellID;
                    pixlist2[no_pixels2].x=x-1;
                    pixlist2[no_pixels2].y=y; 
                    no_pixels2++; 
                }
                
                
                if(y > 0 && B->rptr[x][y-1] < 1 && A->rptr[x][y-1] < level)
                { 
					B->rptr[x][y-1]=cellID; 
					pixlist2[no_pixels2].x=x; 
					pixlist2[no_pixels2].y=y-1; 
					no_pixels2++; 
				}
					
                if(y < mrows-1 && B->rptr[x][y+1] < 1 && A->rptr[x][y+1] < level)
                {
					B->rptr[x][y+1]=cellID; 
					pixlist2[no_pixels2].x=x; 
					pixlist2[no_pixels2].y=y+1;  
					no_pixels2++; 
				}
				
        }
        
        //printf("level: %f, nopixels1 %d and nopixels2 %d\n", level, no_pixels1, no_pixels2);
        
        
        // populating pixellist from old pixellist with allocated pixels only
        no_pixels1 =0;
        
        for (n=0; n < no_pixels2; n++)
    	{
            x=pixlist2[n].x;
            y=pixlist2[n].y;
            
            //look for labels [B->rptr[x][y] > 0]
            if (B->rptr[x][y] > 0)
            {
                pixlist[no_pixels1].x=x;
                pixlist[no_pixels1].y=y;
                no_pixels1++;
            }
            
        } 
    }
    
    
    // ------ post processing of completed segmentation to remove small gaps left over by the initial growing
     for (n=0; n < no_pixels1; n++)
     {
         x=pixlist[n].x;  y=pixlist[n].y;

				if(x < ncols-1 && B->rptr[x+1][y] < 1 && A->rptr[x+1][y] < level) //grow by level again 
                {
                    B->rptr[x+1][y] = B->rptr[x][y]; //growth! new label: B->rptr[x][y]
                }
                if(x > 0 && B->rptr[x-1][y] < 1 && A->rptr[x-1][y] < level)
                {
                    B->rptr[x-1][y]=B->rptr[x][y];
                }
                if(y > 0 && B->rptr[x][y-1] < 1 && A->rptr[x][y-1] < level)
                {
					B->rptr[x][y-1]=B->rptr[x][y]; 
				}
                if(y < mrows-1 && B->rptr[x][y+1] < 1 && A->rptr[x][y+1] < level)
                {
					B->rptr[x][y+1]=B->rptr[x][y]; 
				}
                
        }
        
   
    free( pixlist); free( pixlist2);
    mx_free( A); mx_free(B);
}
