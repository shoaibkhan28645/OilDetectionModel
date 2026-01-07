import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:frontend_flutter/api_service.dart';
import 'package:flutter_animate/flutter_animate.dart';

class DetectScreen extends StatefulWidget {
  const DetectScreen({super.key});

  @override
  State<DetectScreen> createState() => _DetectScreenState();
}

class _DetectScreenState extends State<DetectScreen> {
  File? _image;
  bool _isAnalyzing = false;
  Map<String, dynamic>? _result;
  final ApiService _apiService = ApiService();
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source);
      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          _result = null; // Reset previous result
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error picking image: $e')));
    }
  }

  Future<void> _analyzeImage() async {
    if (_image == null) return;

    setState(() => _isAnalyzing = true);
    try {
      final data = await _apiService.predict(_image!);
      setState(() => _result = data['result']);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString()), backgroundColor: Colors.red),
      );
    } finally {
      setState(() => _isAnalyzing = false);
    }
  }

  void _reset() {
    setState(() {
      _image = null;
      _result = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Image Display Area
          GestureDetector(
            onTap: _image == null
                ? () => _pickImage(ImageSource.gallery)
                : null,
            child: Container(
              height: 300,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.grey[300]!),
                image: _image != null
                    ? DecorationImage(
                        image: FileImage(_image!),
                        fit: BoxFit.cover,
                      )
                    : null,
              ),
              child: _image == null
                  ? Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.add_photo_alternate_outlined,
                          size: 60,
                          color: Colors.grey,
                        ),
                        const SizedBox(height: 10),
                        Text(
                          'Tap to select image',
                          style: TextStyle(color: Colors.grey[600]),
                        ),
                      ],
                    )
                  : Stack(
                      children: [
                        Positioned(
                          top: 10,
                          right: 10,
                          child: IconButton(
                            icon: const Icon(Icons.close, color: Colors.white),
                            style: IconButton.styleFrom(
                              backgroundColor: Colors.black54,
                            ),
                            onPressed: _reset,
                          ),
                        ),
                      ],
                    ),
            ).animate().fadeIn(),
          ),

          const SizedBox(height: 20),

          // Action Buttons
          if (_image == null)
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 15),
                    ),
                  ),
                ),
                const SizedBox(width: 15),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 15),
                    ),
                  ),
                ),
              ],
            ).animate().slideY(begin: 0.5, end: 0),

          if (_image != null && _result == null)
            ElevatedButton(
              onPressed: _isAnalyzing ? null : _analyzeImage,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF00C853),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 15),
              ),
              child: _isAnalyzing
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    )
                  : const Text('Analyze Image', style: TextStyle(fontSize: 18)),
            ).animate().scale(),

          // Result Display
          if (_result != null) _buildResultCard(),
        ],
      ),
    );
  }

  Widget _buildResultCard() {
    final confidence = (_result!['confidence'] as num).toDouble();
    final predictedClass = _result!['predicted_class'] as String;

    // Determine quality/color based on confidence or class
    final color = confidence > 0.8
        ? Colors.green
        : (confidence > 0.5 ? Colors.orange : Colors.red);

    // Example: If class names are "coriander_oil", "mustard_oil"
    // Just display the name prettified
    final displayClass = predictedClass.replaceAll('_', ' ').toUpperCase();

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            const Text(
              'Analysis Result',
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
            const SizedBox(height: 10),
            Text(
              displayClass,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: color,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 10),
            LinearProgressIndicator(
              value: confidence,
              backgroundColor: Colors.grey[200],
              color: color,
              minHeight: 10,
              borderRadius: BorderRadius.circular(5),
            ),
            const SizedBox(height: 10),
            Text(
              'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            const Divider(height: 30),
            // Show probabilities if available
            if (_result!['probabilities'] != null)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: (_result!['probabilities'] as Map<String, dynamic>)
                    .entries
                    .map((e) {
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(e.key.replaceAll('_', ' ')),
                            Text(
                              '${((e.value as num) * 100).toStringAsFixed(1)}%',
                            ),
                          ],
                        ),
                      );
                    })
                    .toList(),
              ),
          ],
        ),
      ),
    ).animate().flipV();
  }
}
