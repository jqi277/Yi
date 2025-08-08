import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(SelfyAIApp());
}

class SelfyAIApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Selfy AI',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: AnalysisPage(),
    );
  }
}

class AnalysisPage extends StatefulWidget {
  @override
  _AnalysisPageState createState() => _AnalysisPageState();
}

class _AnalysisPageState extends State<AnalysisPage> {
  String? selectedExpression;
  String? selectedPosture;
  String? selectedBackground;
  String result = '';

  Future<void> analyzePhoto() async {
    final url = Uri.parse('http://127.0.0.1:8000/analyze/');
    final body = json.encode({
      'expression': selectedExpression,
      'posture': selectedPosture,
      'background': selectedBackground,
    });

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: body,
      );

      if (response.statusCode == 200) {
        // 修复中文乱码问题
        final decoded = json.decode(utf8.decode(response.bodyBytes));
        setState(() {
          result = '''
卦象：${decoded["gua_name"]}
性格解读：${decoded["personality"]}
金钱运势：${decoded["money_fortune"]}
感情运势：${decoded["relationship_fortune"]}
''';
        });
      } else {
        setState(() {
          result = '分析失败，请检查服务是否正常。';
        });
      }
    } catch (e) {
      setState(() {
        result = '网络请求失败：$e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Selfy AI 性格分析')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            DropdownButtonFormField<String>(
              value: selectedExpression,
              hint: Text('选择表情'),
              items: ['calm_controlled', 'happy_smile', 'intense_stare']
                  .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                  .toList(),
              onChanged: (value) => setState(() => selectedExpression = value),
            ),
            DropdownButtonFormField<String>(
              value: selectedPosture,
              hint: Text('选择姿势'),
              items: ['relaxed_leaning', 'confident_pose', 'neutral_sit']
                  .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                  .toList(),
              onChanged: (value) => setState(() => selectedPosture = value),
            ),
            DropdownButtonFormField<String>(
              value: selectedBackground,
              hint: Text('选择背景'),
              items: ['decorative_artsy', 'simple_plain', 'urban_style']
                  .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                  .toList(),
              onChanged: (value) => setState(() => selectedBackground = value),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: analyzePhoto,
              child: Text('分析照片'),
            ),
            SizedBox(height: 20),
            Text(
              result,
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}
