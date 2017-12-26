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
using Eigen::VectorXf;

/**
 * This app demonstrates facial landmark tracking, estimation of the 3D pose
 * and fitting of the shape model of a 3D Morphable Model from a video stream,
 * and merging of the face texture.
 */
int main(int argc, char *argv[])
{
	string modelfile, reference_video, target_video, facedetector, landmarkdetector, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, texturefile;
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
			("reference,r", po::value<string>(&reference_video)->required(),
				"reference video file. If not specified, camera 0 will be used.")
			("target,t", po::value<string>(&target_video)->required(),
				"target video file.")
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

	cv::VideoCapture refcap, tgtcap;
	if (reference_video.empty()) {
		refcap.open(0); // no file given, open the default camera
	}
	else {
		refcap.open(reference_video);
	}
	if (!refcap.isOpened()) {
		cout << "Couldn't open the given file " << reference_video << " or camera 0." << endl;
		return EXIT_FAILURE;
	}
	tgtcap.open(target_video);
	if (!tgtcap.isOpened()) {
		cout << "Couldn't open the given file " << target_video << endl;
		return EXIT_FAILURE;
	}

	vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);

	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);

	cv::namedWindow("reference", 1);
	cv::namedWindow("target", 1);
	cv::namedWindow("result", 1);

	Mat rframe, tframe;

	bool ref_have_face = false;
	bool tgt_have_face = false;
	rcr::LandmarkCollection<cv::Vec2f> ref_lms, tgt_lms;

	// merge all triangles that are facing <60° towards the camera
	// only target one needs texture
	WeightedIsomapAveraging tgt_isomap_averaging(60.f);
	PcaCoefficientMerging ref_pca_shape_merging, tgt_pca_shape_merging;

	vector<float> ref_shape_coefficients, ref_blendshape_coefficients;
	vector<float> tgt_shape_coefficients, tgt_blendshape_coefficients;

	for (;;)
	{
		// get new frames
		refcap >> rframe;
		tgtcap >> tframe;

		if (rframe.empty() || tframe.empty()) { // stop if we're at the end of any of the videos
			break;
		}

		auto DetectLandmarks = [&face_cascade, &rcr_model](bool have_face, rcr::LandmarkCollection<cv::Vec2f> &landmarks, Mat &frame) -> bool
		{
			// do a quick check if the current face's width is <= 50 pixel.
			// If it is, re-initialise the tracking with the face detector.
			if (have_face && get_enclosing_bbox(rcr::to_row(landmarks)).width <= 50) {
				cout << "Reinitialising because the face bounding-box width is <= 50 px" << endl;
				have_face = false;
			}

			if (!have_face) {
				// Run the face detector and obtain the initial estimate using the mean landmarks:
				vector<Rect> detected_faces;
				face_cascade.detectMultiScale(frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
				if (detected_faces.empty()) {
					return false;
				}
				// Rescale the V&J facebox to make it more like an ibug-facebox:
				// (also make sure the bounding box is square, V&J's is square)
				Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);

				landmarks = rcr_model.detect(frame, ibug_facebox);
			}
			else {
				// We already have a face - track and initialise using the enclosing bounding
				// box from the landmarks from the last frame:
				cv::Rect enclosing_bbox;
				enclosing_bbox = get_enclosing_bbox(rcr::to_row(landmarks));
				enclosing_bbox = make_bbox_square(enclosing_bbox);
				landmarks = rcr_model.detect(frame, enclosing_bbox);
			}

			return true;
		};

		ref_have_face = DetectLandmarks(ref_have_face, ref_lms, rframe);
		tgt_have_face = DetectLandmarks(tgt_have_face, tgt_lms, tframe);

		if (!ref_have_face || !tgt_have_face) {
			cv::imshow("reference", rframe);
			cv::imshow("target", tframe);
			cv::waitKey(30);
			continue;
		}

		// Fit the 3DMM:
		vector<Eigen::Vector2f> image_points;
		fitting::RenderingParameters ref_rendering_params, tgt_rendering_params;
		core::Mesh ref_mesh, tgt_mesh;

		// fit reference frame
		std::tie(ref_mesh, ref_rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes, rcr_to_eos_landmark_collection(ref_lms), landmark_mapper, rframe.cols, rframe.rows, edge_topology, ibug_contour, model_contour, 3, 5, 15.0f, std::nullopt, ref_shape_coefficients, ref_blendshape_coefficients, image_points);

		// fit target frame
		std::tie(tgt_mesh, tgt_rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes, rcr_to_eos_landmark_collection(tgt_lms), landmark_mapper, tframe.cols, tframe.rows, edge_topology, ibug_contour, model_contour, 3, 5, 15.0f, std::nullopt, tgt_shape_coefficients, tgt_blendshape_coefficients, image_points);

		// Extract the texture using the fitted mesh from the target frame:
		const Eigen::Matrix<float, 3, 4> affine_cam = fitting::get_3x4_affine_camera_matrix(tgt_rendering_params, tframe.cols, tframe.rows);
		const core::Image4u isomap = render::extract_texture(tgt_mesh, affine_cam, core::from_mat(tframe), true, render::TextureInterpolation::NearestNeighbour, 512);

		// Merge the isomaps - add the current one to the already merged ones:
		Mat merged_isomap = tgt_isomap_averaging.add_and_merge(isomap);

		// transfer the expression coefficients
		VectorXf transferred_shape = morphable_model.get_shape_model().draw_sample(tgt_shape_coefficients) + morphablemodel::to_matrix(blendshapes) * Eigen::Map<const VectorXf>(ref_blendshape_coefficients.data(), ref_blendshape_coefficients.size());

		// Now current shape coefficients are already used, then we merge the shape coefficients
		ref_shape_coefficients = ref_pca_shape_merging.add_and_merge(ref_shape_coefficients);
		tgt_shape_coefficients = tgt_pca_shape_merging.add_and_merge(tgt_shape_coefficients);

		// generate transferred mesh
		core::Mesh transferred_mesh = morphablemodel::sample_to_mesh(transferred_shape, morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());

		// Render the transferred model
		core::Image4u rendering;
		std::tie(rendering, std::ignore) = render::render(transferred_mesh, tgt_rendering_params.get_modelview(), tgt_rendering_params.get_projection(),
														  tgt_rendering_params.get_screen_width(), tgt_rendering_params.get_screen_height(),
														  render::create_mipmapped_texture(merged_isomap), true, false, false);

		// map the rendered image to the target frame
		cv::Mat rendered_face(rendering.rows, rendering.cols, CV_8UC3);
		for (int col = 0; col < rendering.cols; ++col)
		{
			for (int row = 0; row < rendering.rows; ++row)
			{
				auto r = rendering(row, col)[0];
				auto g = rendering(row, col)[1];
				auto b = rendering(row, col)[2];
				auto a = rendering(row, col)[3];

				// if the rendered pixel is visible, use it,
				// otherwise, use the correspondence pixel in tframe
				if (r|g|b) {
					rendered_face.at<cv::Vec3b>(row, col) = cv::Vec3b(r, g, b);
				}
				else {
					rendered_face.at<cv::Vec3b>(row, col) = tframe.at<cv::Vec3b>(row, col);
				}
			}
		}

		cv::imshow("reference", rframe);
		cv::imshow("target", tframe);
		cv::imshow("result", rendered_face);

		auto key = cv::waitKey(30);
		if (key == 'q') break;
		if (key == 'r') {
			ref_have_face = false;
			tgt_have_face = false;
			tgt_isomap_averaging = WeightedIsomapAveraging(60.f);
		}
		if (key == 's') {
			// save an obj + current merged isomap to the disk:
			core::Mesh neutral_expression = morphablemodel::sample_to_mesh(morphable_model.get_shape_model().draw_sample(tgt_shape_coefficients), morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
			core::write_textured_obj(neutral_expression, "current_merged.obj");
			cv::imwrite("current_merged.isomap.png", merged_isomap);
		}

	} // end for loop

	return EXIT_SUCCESS;
};
