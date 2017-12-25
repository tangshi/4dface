/*
 * 4dface: Real-time 3D face tracking and reconstruction from 2D video.
 *
 * File: apps/4dface.cpp
 *
 * Copyright 2015, 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "helpers.hpp"
#include "guidedfilter.h"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Image_opencv_interop.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/render.hpp"
#include "eos/render/texture_extraction.hpp"

#include "rcr/model.hpp"
#include "cereal/cereal.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"

#include "Eigen/Core"
#include "Eigen/Dense"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Rect;
using std::cout;
using std::endl;
using std::vector;
using std::string;


/**
 * This app demonstrates facial landmark tracking, estimation of the 3D pose
 * and fitting of the shape model of a 3D Morphable Model from a video stream,
 * and merging of the face texture.
 */
int main(int argc, char *argv[])
{
	string modelfile, inputvideo, facedetector, landmarkdetector, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, texturefile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("morphablemodel,m", po::value<string>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				"a Morphable Model stored as cereal BinaryArchive")
			("facedetector,f", po::value<string>(&facedetector)->required()->default_value("../share/haarcascade_frontalface_alt2.xml"),
				"full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
			("landmarkdetector,l", po::value<string>(&landmarkdetector)->required()->default_value("../share/face_landmarks_model_rcr_68.bin"),
				"learned landmark detection model")
			("mapping,p", po::value<string>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
				"landmark identifier to model vertex number mapping")
			("model-contour,c", po::value<string>(&contourfile)->required()->default_value("../share/sfm_model_contours.json"),
				"file with model contour indices")
			("edge-topology,e", po::value<string>(&edgetopologyfile)->required()->default_value("../share/sfm_3448_edge_topology.json"),
				"file with model's precomputed edge topology")
			("blendshapes,b", po::value<string>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
				"file with blendshapes")
			("texture,t", po::value<string>(&texturefile)->required()->default_value("../share/makeup.png"),
				"image file used as texture")
			("input,i", po::value<string>(&inputvideo),
				"input video file. If not specified, camera 0 will be used.")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: 4dface [options]" << endl;
			cout << desc;
			return EXIT_FAILURE;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

	// Load the Morphable Model and the LandmarkMapper:
	morphablemodel::MorphableModel morphable_model = morphablemodel::load_model(modelfile);
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);
	fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile);

	rcr::detection_model rcr_model;
	// Load the landmark detection model:
	try {
		rcr_model = rcr::load_detection_model(landmarkdetector);
	}
	catch (const cereal::Exception& e) {
		cout << "Error reading the RCR model " << landmarkdetector << ": " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the face detector from OpenCV:
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(facedetector))
	{
		cout << "Error loading the face detector " << facedetector << "." << endl;
		return EXIT_FAILURE;
	}

	cv::VideoCapture cap;
	if (inputvideo.empty()) {
		cap.open(0); // no file given, open the default camera
	}
	else {
		cap.open(inputvideo);
	}
	if (!cap.isOpened()) {
		cout << "Couldn't open the given file or camera 0." << endl;
		return EXIT_FAILURE;
	}

	vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);

	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);

	cv::namedWindow("video", 1);
	cv::namedWindow("render", 1);

	Mat frame, unmodified_frame, isomap, mask;

	isomap = cv::imread(texturefile, CV_LOAD_IMAGE_UNCHANGED);

	render::Texture texture = render::create_mipmapped_texture(isomap);

	bool have_face = false;
	rcr::LandmarkCollection<cv::Vec2f> current_landmarks;
	Rect current_facebox;
	PcaCoefficientMerging pca_shape_merging;
	vector<float> shape_coefficients, blendshape_coefficients;
	vector<Eigen::Vector2f> image_points;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty()) { // stop if we're at the end of the video
			break;
		}

		// We do a quick check if the current face's width is <= 50 pixel. If it is, we re-initialise the tracking with the face detector.
		if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
			cout << "Reinitialising because the face bounding-box width is <= 50 px" << endl;
			have_face = false;
		}

		unmodified_frame = frame.clone();
		vector<Rect> detected_faces;
		cv::Rect enclosing_bbox;
		if (!have_face) {
			// Run the face detector and obtain the initial estimate using the mean landmarks:

			face_cascade.detectMultiScale(unmodified_frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
			if (detected_faces.empty()) {
				cv::imshow("video", frame);
				cv::waitKey(30);
				continue;
			}
			cv::rectangle(frame, detected_faces[0], { 255, 0, 0 });
			// Rescale the V&J facebox to make it more like an ibug-facebox:
			// (also make sure the bounding box is square, V&J's is square)
			Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);

			current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);

			have_face = true;
		}
		else {
			// We already have a face - track and initialise using the enclosing bounding
			// box from the landmarks from the last frame:
			enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
			enclosing_bbox = make_bbox_square(enclosing_bbox);
			current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
		}

		// Fit the 3DMM:
		fitting::RenderingParameters rendering_params;
		core::Mesh mesh;
		std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes, rcr_to_eos_landmark_collection(current_landmarks), landmark_mapper, unmodified_frame.cols, unmodified_frame.rows, edge_topology, ibug_contour, model_contour, 3, 5, 15.0f, std::nullopt, shape_coefficients, blendshape_coefficients, image_points);

		core::Image4u rendering;
		std::tie(rendering, std::ignore) = render::render(mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
														  rendering_params.get_screen_width(), rendering_params.get_screen_height(),
														  texture, false, false, false);

		int x0 = rendering.cols;
		int y0 = rendering.rows;
		int x1 = 0;
		int y1 = 0;

		cv::Mat rendered_face(rendering.rows, rendering.cols, CV_8UC3);
		for (int col = 0; col < rendering.cols; ++col)
		{
			for (int row = 0; row < rendering.rows; ++row)
			{
				auto r = rendering(row, col)[0];
				auto g = rendering(row, col)[1];
				auto b = rendering(row, col)[2];
				auto a = rendering(row, col)[3];

				// convert rendering(Image4u) into opencv Mat type
				rendered_face.at<cv::Vec3b>(row, col) = cv::Vec3b(r, g, b);

				// the alpha value of a pixel is initially 0, and changed to 255 after rendered
				if (a)
				{
					// replace pixel value with the rendering result
					frame.at<cv::Vec3b>(row, col) = cv::Vec3b(r, g, b);

					// store the border of face rectangle
					if (row < y0) { y0 = row; }
					if (row > y1) { y1 = row; }
					if (col < x0) { x0 = col; }
					if (col > x1) { x1 = col; }
				}
			}
		}

		Rect rendered_rect(x0, y0, x1 - x0, y1 - y0);
		Mat face_area(rendered_face, rendered_rect);

		Mat face_im = Mat::zeros(rendered_rect.size(), CV_8UC3);
		Mat(unmodified_frame, rendered_rect).copyTo(face_im, face_area);

		Mat render_im = Mat::zeros(rendered_rect.size(), CV_8UC3);
		Mat(frame, rendered_rect).copyTo(render_im, face_area);

		face_im.convertTo(face_im, CV_32F, 1.0 / 255.0);
		render_im.convertTo(render_im, CV_32F, 1.0 / 255.0);

		int r = 16;
		double eps = 0.1 * 0.1;

		Mat base = guidedFilter(face_im, face_im, r, eps);

		// S = (face_im ./ base) ^ 2.2;
		cv::divide(face_im, base, face_im);
		cv::pow(face_im, 2.2, face_im);

		// R = 0.4 * base + 0.6 * render_im;
		// X = R .* S;
		cv::addWeighted(base, 0.4, render_im, 0.6, 0, render_im);
		cv::multiply(render_im, face_im, render_im);

		render_im.convertTo(render_im, CV_8U, 255);
		render_im.copyTo(Mat(frame, rendered_rect), face_area);

		shape_coefficients = pca_shape_merging.add_and_merge(shape_coefficients);

		cv::imshow("video", unmodified_frame);
		cv::imshow("render", frame);

		auto key = cv::waitKey(30);
		if (key == 'q') break;
		if (key == 's')
		{
			Mat face_im = Mat(unmodified_frame, rendered_rect);
			cv::imwrite("face.png", face_im);

			Mat render_im = Mat(frame, rendered_rect);
			cv::imwrite("render.png", render_im);

			// save an obj to the disk:
			core::write_textured_obj(mesh, "current_mesh.obj");
		}
	}

	return EXIT_SUCCESS;
};
