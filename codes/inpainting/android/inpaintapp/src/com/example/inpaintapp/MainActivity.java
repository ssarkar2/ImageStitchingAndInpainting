package com.example.inpaintapp;
import static java.lang.Math.round;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.FileChannel;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup.LayoutParams;
import android.view.ViewTreeObserver;
import android.widget.ImageView;
import android.widget.Toast;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

//notes: make the rectangle selection better
//verify if coordinates saved wrt original are correct. (originalRectX originalRectY originalRectWidth originalRectHeight)

public class MainActivity extends Activity {
	public native int inPaint(String inpaintRect, String orig, String maskpath, int originalRectX, int originalRectY, int originalRectWidth, int originalRectHeight);
	
	private MenuItem mPickImage, mReset;
	
	private int originalImageWidth, originalImageHeight; //dimensions of original iage
	private int originalRectX = 0, originalRectY = 0, originalRectWidth = 0, originalRectHeight = 0;  //location and dimensions of the selected rectangle wrt original image
	private float heightImgView = 0, widthImgView = 0;
	//int[] viewCoords = new int[2];
	//int windowwidth ,  windowheight;
	
	private ImageView image;
	private String fileName_orig;
	private String testpath, scaledpath, maskpath;
	
	private float sxtemp=0, sytemp=0, sx=0, sy=0, ex=0, ey=0; //start and end coordinates
	private long startTime = 0, stopTime = 0, thresholdTime = 400;
	private boolean rectanglePresent = false;
	
	private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OPENCV", "OpenCV loaded successfully");
                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("inpaintapp");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
	
    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        image = (ImageView) findViewById(R.id.imageView1);
        //image.setImageResource(R.drawable.test2);
        
        //String path = Environment.getExternalStorageDirectory().toString() +  "/44.png";   
        
        //Bitmap bmp = BitmapFactory.decodeFile(path);
        //Bitmap scaledBitmap = BitmapFactory.decodeFile(path);
        //image.setImageBitmap(scaledBitmap);   
    }

    @Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data) 
	{
    	//find image view dimensions
    	//heightImgView = ((ImageView) findViewById(R.id.imageView1)).getHeight();
    	//widthImgView = ((ImageView) findViewById(R.id.imageView1)).getWidth();
    	ImageView iv=(ImageView)findViewById(R.id.imageView1);
    	 ViewTreeObserver vto = iv.getViewTreeObserver();
    	    vto.addOnPreDrawListener(new ViewTreeObserver.OnPreDrawListener() {
    	        public boolean onPreDraw() {
    	        	ImageView iv=(ImageView)findViewById(R.id.imageView1);
    	        	heightImgView = iv.getMeasuredHeight();
    	        	widthImgView = iv.getMeasuredWidth();
    	        	//cornerRow = iv.getTop();
    	        	//cornerCol = iv.getLeft();

    	        	//iv.getLocationOnScreen(viewCoords);
    	            return true;
    	        }
    	    });

    	
	    if(resultCode==RESULT_CANCELED)
	    {
	        // action cancelled
	    }
	    if(resultCode==RESULT_OK)
	    {
	        Uri selectedimg = data.getData();
	        	       
	        
	        String scheme = selectedimg.getScheme();
	        if (scheme.equals("file")) 
	        {
	        	fileName_orig = selectedimg.getLastPathSegment();
	        	//Log.e("filename", fileName_orig);
	        }
	        else
	        {
	        	String[] proj = { MediaStore.Images.Media.DATA };
	        	//Cursor cursor = managedQuery(selectedimg, proj, null, null, null);
	        	Cursor cursor = getContentResolver().query(selectedimg, proj, null, null, null);
	        	int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
	            cursor.moveToFirst();
	            fileName_orig = cursor.getString(column_index);
	            //Log.e("filename2", fileName_orig); 
	            
	            if (cursor != null) {
	                cursor.close();
	            }
	        }
	        
	        //create copy of selected image
	        try{
	        	String root = Environment.getExternalStorageDirectory().getAbsolutePath()+"/";
	            File createDir = new File(root+"631folder"+File.separator);
	            if(!createDir.exists()) {
	                createDir.mkdir();
	            }
	            String substr = fileName_orig.substring(fileName_orig.length() - 3);  //extension of the file
	            File file = new File(root + "631folder" + File.separator +"test." + substr);
	            File file1 = new File(root + "631folder" + File.separator +"scaled." + substr);
	            testpath = root + "631folder" + File.separator +"test." + substr;
	            scaledpath = root + "631folder" + File.separator +"scaled." + substr;
	            maskpath = root + "631folder" + File.separator +"mask." + substr;
	            try {
	            	file.createNewFile();
                    copyFile(new File(fileName_orig), file);
                    file1.createNewFile();
                    copyFile(new File(fileName_orig), file1);
                } catch (IOException e) {e.printStackTrace();}
	        }catch(Exception e){}
	        
	        //set the copied image in the image view
	        try{
	        	Bitmap bmp = BitmapFactory.decodeFile(testpath);
	        	originalImageWidth = bmp.getWidth();
	        	originalImageHeight = bmp.getHeight();
	        	Bitmap scaledBitmap = BitmapFactory.decodeFile(testpath);
	        	//image.setImageBitmap(MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedimg));
	        	image.setImageBitmap(scaledBitmap);
	        	
	        	//windowwidth = getWindowManager().getDefaultDisplay().getWidth();
	            //windowheight = getWindowManager().getDefaultDisplay().getHeight();
	        	image.setOnTouchListener(new View.OnTouchListener() {
                    @Override
                    public boolean onTouch(View v, MotionEvent event) {
                          LayoutParams layoutParams = (LayoutParams) image.getLayoutParams();
                          float screenX, screenY;
                          switch (event.getAction()) {
                          case (MotionEvent.ACTION_UP) :
              				stopTime = System.currentTimeMillis();
              				if (stopTime - startTime > thresholdTime)
              				{
              					screenX = event.getX();
                        	  	screenY = event.getY();
                        	  	float tempx = screenX-v.getLeft();
                  				float tempy = screenY-v.getTop();
                        	  	if (tempx > 0 && tempy > 0 && screenX < v.getRight() && screenY < v.getBottom())
                        	  	{
                        	  		//save rectangle coordinates and call rectangle painter
                        	  		if (sxtemp < tempx)
                        	  		{
                        	  			sx = sxtemp;
                        	  			ex = tempx;
                        	  		}
                        	  		else
                        	  		{
                        	  			sx = tempx;
                        	  			ex = sxtemp;
                        	  		}
                        	  		if (sytemp < tempy)
                        	  		{
                        	  			sy = sytemp;
                        	  			ey = tempy;
                        	  		}
                        	  		else
                        	  		{
                        	  			sy = tempy;
                        	  			ey = sytemp;
                        	  		}

                        	  		rectanglePresent = true;
                        	  		
                        	  		Bitmap bmp = BitmapFactory.decodeFile(scaledpath);
                        	  		int bitmapwidth = bmp.getWidth();  int bitmapheight = bmp.getHeight();
                        	  		
                        	  		//save location of inpaining rectangle wrt original image
                        	  		//need to verify if these are correct
                        	  		originalRectWidth = round(((ex-sx)/widthImgView) * bitmapwidth);
                        	  		originalRectHeight = round(((ey-sy)/heightImgView) * bitmapheight);
                        	  		originalRectX = originalRectX + round((sx/widthImgView) * bitmapwidth);
                        	  		originalRectY = originalRectY + round((sy/heightImgView) * bitmapheight);
	        						
                        	  		//Log.e("bitmap", Integer.toString(round((sx/widthImgView) * bitmapwidth)) + " " +Integer.toString(round((sy/heightImgView) * bitmapheight)) + " " +Integer.toString(round(((ex-sx)/widthImgView) * bitmapwidth)) + " " +Integer.toString(round(((ey-sy)/heightImgView) * bitmapheight)));
                        	  		Bitmap newBitmap = Bitmap.createBitmap(bmp, round((sx/widthImgView) * bitmapwidth), round((sy/heightImgView) * bitmapheight), round(((ex-sx)/widthImgView) * bitmapwidth), round(((ey-sy)/heightImgView) * bitmapheight));
                        	  		image = (ImageView) findViewById(R.id.imageView1);
                        	  		image.setImageBitmap(newBitmap);
                        	  		
                        	  		try {
                        	  			OutputStream stream = new FileOutputStream(scaledpath);
                        	  			newBitmap.compress(CompressFormat.JPEG, 100, stream);
                        	  		}catch(Exception e){}
                        	  		
                        	  		Toast.makeText(getApplicationContext(), "Rectangle set", Toast.LENGTH_SHORT).show();
                        	  		
                        	  	}
                        	  	else
                        	  	{
                        	  		Toast.makeText(getApplicationContext(), "Invalid rectangle endpoint", Toast.LENGTH_SHORT).show();
                        	  	}
              				}
              				else
            				{
            					if (rectanglePresent == true)
            					{
            						Toast.makeText(getApplicationContext(), "Inpainting", Toast.LENGTH_SHORT).show();
            						//start inpainting	   
            						inPaint(scaledpath, testpath, maskpath, originalRectX, originalRectY, originalRectWidth, originalRectHeight);
            						
            						Bitmap inpainted = BitmapFactory.decodeFile(testpath);
            			        	image.setImageBitmap(inpainted);
            			        	
            						rectanglePresent = false;
            					}
            					else
            					{
            						Toast.makeText(getApplicationContext(), "No rectangle found", Toast.LENGTH_SHORT).show();
            					}
            				}
              				break;
                          case MotionEvent.ACTION_DOWN:
                        	  	screenX = event.getX();
                        	  	screenY = event.getY();
                        	  	sxtemp = screenX-v.getLeft();
                  				sytemp = screenY-v.getTop();
                  				if (sxtemp > 0 && sytemp > 0 && screenX < v.getRight() && screenY < v.getBottom())
                  					startTime = System.currentTimeMillis();
                  				else
                  					Toast.makeText(getApplicationContext(), "Invalid rectangle startpoint", Toast.LENGTH_SHORT).show();
                  				break;
                          //case MotionEvent.ACTION_MOVE:
                                 //break;
                          case (MotionEvent.ACTION_CANCEL) :
                  				Log.d("touchtype","Action was CANCEL");
                  				break;
                  		  case (MotionEvent.ACTION_OUTSIDE) :
                  			  Log.d("touchtype","Movement occurred outside bounds " + "of current screen element");
                  				break;
                          default:
                                 break;
                          }
                          return true;
                    }
             });
	        }
	        catch(Exception e){}
	    }
	}

    public boolean onOptionsItemSelected(MenuItem item) {
    	
    	if (item == mPickImage){
    		Intent intent = new Intent();  
    		intent.setType("image/*");  
    		intent.setAction(Intent.ACTION_GET_CONTENT);  
    		startActivityForResult(Intent.createChooser(intent, "Choose Picture"), 1);
    	} else if(item == mReset) {
    		Bitmap bmp = BitmapFactory.decodeFile(testpath);
        	originalImageWidth = bmp.getWidth();
        	originalImageHeight = bmp.getHeight();
        	Bitmap scaledBitmap = BitmapFactory.decodeFile(testpath);
        	//image.setImageBitmap(MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedimg));
        	image.setImageBitmap(scaledBitmap);
        	
        	try {
	  			OutputStream stream = new FileOutputStream(scaledpath);
	  			scaledBitmap.compress(CompressFormat.JPEG, 100, stream);
	  		}catch(Exception e){}
        	
	  		rectanglePresent = false;
	  		originalRectX = 0; originalRectY = 0; originalRectWidth = 0; originalRectHeight = 0;	
	  		sxtemp=0; sytemp=0; sx=0; sy=0; ex=0; ey=0; //start and end coordinates
	  		startTime = 0; stopTime = 0;
    	}

    	return true;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        //getMenuInflater().inflate(R.menu.main, menu);

    	mPickImage = menu.add("Pick Image");
    	mReset = menu.add("Reset");
    	
        return true;
    }
    
    private void copyFile(File sourceFile, File destFile) throws IOException {
        if (!sourceFile.exists()) {
            return;
        }

        FileChannel source = null;
            FileChannel destination = null;
            source = new FileInputStream(sourceFile).getChannel();
            destination = new FileOutputStream(destFile).getChannel();
            if (destination != null && source != null) {
                destination.transferFrom(source, 0, source.size());
            }
            if (source != null) {
                source.close();
            }
            if (destination != null) {
                destination.close();
            }
    }
}
