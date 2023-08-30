import React from 'react'
import { useEffect, useState } from 'react'
import { Document, Page, Text, StyleSheet, View, PDFViewer } from '@react-pdf/renderer'
import { parseCookies } from 'nookies'


const styles = StyleSheet.create({
  page: {
    flexDirection: 'column',
    backgroundColor: '#ffffff',
    padding: '20pt',
  },
  section: {
    marginBottom: '10pt',
  },
  title: {
    fontSize: '18pt',
    fontWeight: 'bold',
    // marginBottom: '10pt',
    textAlign: 'center',
  },
  subject: {
    fontSize: '14pt',
    marginBottom: '10pt',
    textAlign: 'center',
  },
  marks: {
    fontSize: '14pt',
    marginBottom: '10pt',
    textAlign: 'center',
    flexDirection: 'column',
  },
  question: {
    fontSize: '12pt',
    marginBottom: '10pt',
    marginTop: '10pt',
  },
  questionMarks: {
    fontSize: '12pt',
    textAlign: 'right',
    marginBottom: '10pt',
    marginTop: '10pt',
    marginLeft: 'auto',
  },
  flex: {
    display: 'flex',
    flexDirection: 'row',
    // justifyContent: 'space-between',
  },
  questionType: {
    fontSize: '13pt',
    marginBottom: '10pt',
    marginTop: '10pt',
  },
  time: {
    fontSize: '13pt',
    marginBottom: '10pt',
    marginTop: '10pt',
    textAlign: 'right',
  }
})
export default function Pdfgen({ title, data, marks }) {
  // console.log(data, title)
  const [questions, setQuestions] = useState([])
  useEffect(()=>{
    console.log(data)
    setQuestions(data)
    console.log(questions)
  },[questions])
  return (
    <Document>
      <Page style={styles.page}>
        <View style={styles.section}>
          <Text style={styles.title}>{title}</Text>
          <Text style={styles.subject}>Subject : Computer Science</Text>
          <Text style={styles.marks}>Total Marks: {marks}</Text>
          {/* <Text style={styles.time}>Time: {time && timeinminhrs(time)}</Text> */}
        </View>
        {/* ***********ei section ta map korte hobe*********** */}
        {/* <View style={styles.section}>
          {data&&data.map((question, index) => (
            <View key={index} style={styles.section}>
              <View style={styles.flex}>
                <Text style={styles.question}>{index + 1}. {question.question}</Text>
                <Text style={styles.questionMarks}>{question.marks}</Text>
              </View>
              <Text style={styles.questionType}>{question.type}</Text>
            </View>
          ))
          }
        </View> */}
        
      </Page>
    </Document>
  )
}
