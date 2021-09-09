#include "OpenCV_Reader.h"

/** Function Headers */

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

// Sets default values
AOpenCV_Reader::AOpenCV_Reader(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {
	PrimaryActorTick.bCanEverTick = true;
	// ensure the root component exists
	if (!RootComponent)
		RootComponent = CreateDefaultSubobject<USceneComponent>("Root");
	FAttachmentTransformRules rules = FAttachmentTransformRules(EAttachmentRule::KeepRelative, false);
	Screen_Raw = CreateDefaultSubobject<UStaticMeshComponent>("Screen Raw");
	Screen_Raw->AttachToComponent(RootComponent, rules);
	Screen_Post = CreateDefaultSubobject<UStaticMeshComponent>("Screen Post");
	Screen_Post->AttachToComponent(RootComponent, rules);
	Brightness = 0;
	Multiply = 1;
	// Initialize OpenCV and webcam properties
	CameraID = 0;
	VideoTrackID = 0;
	isStreamOpen = false;
	VideoSize = FVector2D(1920, 1080);
	RefreshRate = 30.0f;
	

}

// Called when the game starts or when spawned
void AOpenCV_Reader::BeginPlay() {
	Super::BeginPlay();
	isStreamOpen = true;
	// Prepare the color data array
	ColorData.AddDefaulted(VideoSize.X * VideoSize.Y);

	// setup openCV
	cvSize = cv::Size(VideoSize.X, VideoSize.Y);
	cvMat = cv::Mat(cvSize, CV_8UC4, ColorData.GetData());
	LoadOpenCV();

	// create dynamic texture
	Camera_Texture2D = UTexture2D::CreateTransient(VideoSize.X, VideoSize.Y, PF_B8G8R8A8);
#if WITH_EDITORONLY_DATA
	Camera_Texture2D->MipGenSettings = TMGS_NoMipmaps;
#endif
	Camera_Texture2D->SRGB = Camera_RenderTarget->SRGB;

    cv::namedWindow(main_window_name, cv::WINDOW_NORMAL);
    cv::moveWindow(main_window_name, 400, 100);
    cv::namedWindow(face_window_name, cv::WINDOW_NORMAL);
    cv::moveWindow(face_window_name, 10, 100);
    cv::namedWindow("Right Eye", cv::WINDOW_NORMAL);
    cv::moveWindow("Right Eye", 10, 600);
    cv::namedWindow("Left Eye", cv::WINDOW_NORMAL);
    cv::moveWindow("Left Eye", 10, 800);
    ellipse(skinCrCbHist, cv::Point(113, 155), cv::Size(23, 15),
        43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
}

//tick function that reads camera
void AOpenCV_Reader::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	RefreshTimer += DeltaTime;
	if (isStreamOpen && RefreshTimer >= 1.0f / RefreshRate) {
		RefreshTimer -= 1.0f / RefreshRate;
		ReadFrame();
		OnNextVideoFrame();
	}
}

//reads the camera and locate eyes 
void AOpenCV_Reader::ReadFrame() {
	if (!Camera_Texture2D || !Camera_RenderTarget) return;
	// Read the pixels from the RenderTarget and store them in a FColor array
	FRenderTarget* RenderTarget = Camera_RenderTarget->GameThread_GetRenderTargetResource();
	RenderTarget->ReadPixels(ColorData);
	// Get the color data
	cvMat = cv::Mat(cvSize, CV_8UC4, ColorData.GetData());
	cvMat.convertTo(cvMat, -1, Multiply, Brightness);

    //***NEW EYE TRACKING CODE***
    createCornerKernels();

    if (!cvMat.empty()) {

        // mirror it
        cv::flip(cvMat, cvMat, 1);
        cvMat.copyTo(debugImage);

        // Apply the classifier to the frame
        if (!cvMat.empty()) {
            detectAndDisplay(cvMat, cvFaceCascade);
        }
        else {
            printf(" --(!) No captured frame -- Break!");
        }

        imshow(main_window_name, debugImage);
    }

    releaseCornerKernels();
    //***NEW EYE TRACKING CODE***

	// Lock the texture so we can read / write to it
	void* TextureData = Camera_Texture2D->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
	const int32 TextureDataSize = ColorData.Num() * 4;
	// set the texture data
	FMemory::Memcpy(TextureData, ColorData.GetData(), TextureDataSize);
	// Unlock the texture
	Camera_Texture2D->PlatformData->Mips[0].BulkData.Unlock();
	// Apply Texture changes to GPU memory
	Camera_Texture2D->UpdateResource();
	
}

void AOpenCV_Reader::LoadOpenCV() {
	// setup openCV
	FString ProjectPath = FPaths::ConvertRelativePathToFull(FPaths::ProjectDir());

	Path_Cascade_Face = ProjectPath + Path_Cascade_Face;
	Path_Cascade_Eyes = ProjectPath + Path_Cascade_Eyes;

	cvFaceCascade = cv::CascadeClassifier(TCHAR_TO_UTF8(*Path_Cascade_Face));
	cvEyeCascade = cv::CascadeClassifier(TCHAR_TO_UTF8(*Path_Cascade_Eyes));

	// load cascades
	if (!cvFaceCascade.load(TCHAR_TO_UTF8(*Path_Cascade_Face)))
	{
		UE_LOG(LogTemp, Error, TEXT("Error loading face cascade\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *Path_Cascade_Face);
	};
	if (!cvEyeCascade.load(TCHAR_TO_UTF8(*Path_Cascade_Eyes)))
	{
		UE_LOG(LogTemp, Error, TEXT("Error loading eyes cascade\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *Path_Cascade_Eyes);
	};
}

//find eyes
void AOpenCV_Reader::findEyes(cv::Mat frame_gray, cv::Rect face) {
    cv::Mat faceROI = frame_gray(face);
    cv::Mat debugFace = faceROI;

    if (kSmoothFaceImage) {
        double sigma = kSmoothFaceFactor * face.width;
        GaussianBlur(faceROI, faceROI, cv::Size(0, 0), sigma);
    }
    //-- Find eye regions and draw them
    int eye_region_width = face.width * (kEyePercentWidth / 100.0);
    int eye_region_height = face.width * (kEyePercentHeight / 100.0);
    int eye_region_top = face.height * (kEyePercentTop / 100.0);
    cv::Rect leftEyeRegion(face.width * (kEyePercentSide / 100.0),
        eye_region_top, eye_region_width, eye_region_height);
    cv::Rect rightEyeRegion(face.width - eye_region_width - face.width * (kEyePercentSide / 100.0),
        eye_region_top, eye_region_width, eye_region_height);

    //-- Find Eye Centers
    cv::Point leftPupil = findEyeCenter(faceROI, leftEyeRegion, "Left Eye");
    cv::Point rightPupil = findEyeCenter(faceROI, rightEyeRegion, "Right Eye");
    // get corner regions
    cv::Rect leftRightCornerRegion(leftEyeRegion);
    leftRightCornerRegion.width -= leftPupil.x;
    leftRightCornerRegion.x += leftPupil.x;
    leftRightCornerRegion.height /= 2;
    leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
    cv::Rect leftLeftCornerRegion(leftEyeRegion);
    leftLeftCornerRegion.width = leftPupil.x;
    leftLeftCornerRegion.height /= 2;
    leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
    cv::Rect rightLeftCornerRegion(rightEyeRegion);
    rightLeftCornerRegion.width = rightPupil.x;
    rightLeftCornerRegion.height /= 2;
    rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
    cv::Rect rightRightCornerRegion(rightEyeRegion);
    rightRightCornerRegion.width -= rightPupil.x;
    rightRightCornerRegion.x += rightPupil.x;
    rightRightCornerRegion.height /= 2;
    rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
    rectangle(debugFace, leftRightCornerRegion, 200);
    rectangle(debugFace, leftLeftCornerRegion, 200);
    rectangle(debugFace, rightLeftCornerRegion, 200);
    rectangle(debugFace, rightRightCornerRegion, 200);
    // change eye centers to face coordinates
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;

    //Unreal variables for ball rolling
    rightEyeWidth = (rightEyeRegion.x + eye_region_width) - rightEyeRegion.x;
    leftEyeWidth = (leftEyeRegion.x + eye_region_width) - leftEyeRegion.x;
    rightEyeHeight = (rightEyeRegion.y + eye_region_height) - rightEyeRegion.y;
    leftEyeHeight = (leftEyeRegion.y + eye_region_height) - leftEyeRegion.y;
    rightXMidPoint = rightEyeRegion.x + (rightEyeWidth/2);
    leftXMidPoint = leftEyeRegion.x + (leftEyeWidth / 2);
    rightYMidPoint = rightEyeRegion.y + (rightEyeHeight / 2);
    leftYMidPoint = leftEyeRegion.y + (leftEyeHeight / 2);
    leftPupilX = leftPupil.x;
    leftPupilY = leftPupil.y;
    rightPupilX = rightPupil.x;
    rightPupilY = rightPupil.y;

    // draw eye centers
    circle(debugFace, rightPupil, 3, 1234);
    circle(debugFace, leftPupil, 3, 1234);

    //-- Find Eye Corners
    if (kEnableEyeCorner) {
        cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
        leftRightCorner.x += leftRightCornerRegion.x;
        leftRightCorner.y += leftRightCornerRegion.y;
        cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
        leftLeftCorner.x += leftLeftCornerRegion.x;
        leftLeftCorner.y += leftLeftCornerRegion.y;
        cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
        rightLeftCorner.x += rightLeftCornerRegion.x;
        rightLeftCorner.y += rightLeftCornerRegion.y;
        cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
        rightRightCorner.x += rightRightCornerRegion.x;
        rightRightCorner.y += rightRightCornerRegion.y;
        circle(faceROI, leftRightCorner, 3, 200);
        circle(faceROI, leftLeftCorner, 3, 200);
        circle(faceROI, rightLeftCorner, 3, 200);
        circle(faceROI, rightRightCorner, 3, 200);

        UE_LOG(LogTemp, Warning, TEXT("rightRightCorner: %d\n"), rightRightCorner.x);
        UE_LOG(LogTemp, Warning, TEXT("rightLeftCorner: %d\n"), rightLeftCorner.x);
    }



    imshow(face_window_name, faceROI);
}


//find skin of face
cv::Mat findSkin(cv::Mat& frame) {
    cv::Mat input;
    cv::Mat output = cv::Mat(frame.rows, frame.cols, CV_8U);

    cvtColor(frame, input, cv::COLOR_BGR2YCrCb);

    for (int y = 0; y < input.rows; ++y) {
        const cv::Vec3b* Mr = input.ptr<cv::Vec3b>(y);
        //    uchar *Or = output.ptr<uchar>(y);
        cv::Vec3b* Or = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input.cols; ++x) {
            cv::Vec3b ycrcb = Mr[x];
            //      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
                Or[x] = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return output;
}

/**
 * @function detectAndDisplay
 */
//detect face
void AOpenCV_Reader::detectAndDisplay(cv::Mat frame, cv::CascadeClassifier cascade) {

    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY); // convert image to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast single c

    //-- Detect faces
    cascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE | cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(150, 150));

    if(faces.size()>0)
        for (int i = 0; i < faces.size(); i++)
        {
            rectangle(debugImage, faces[i], 1234);
        }
        //-- Show what you got
        if (faces.size() > 0) {
            findEyes(grayscale, faces[0]);
        }
}