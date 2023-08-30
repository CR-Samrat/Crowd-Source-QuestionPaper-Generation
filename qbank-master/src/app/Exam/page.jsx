'use client'
import React,{useState,useEffect} from 'react'
import MCQExam from './MCQExam'
import {parseCookies} from 'nookies'
export default function page() {
  const [startExam, setStartExam] = useState(false);
  const {expert, email} = parseCookies();
  useEffect(() => {
    if(!email){
      window.location.href='/Signin'
    }
    if(expert!='-1'){
      window.location.href='/'
    }
  }, [])

  return (
    <div>{!startExam?<div>
      <div className="container mx-auto mt-4 p-4">
      <h1 className="text-2xl font-semibold mb-4 text-center">You are not a subject matter expert</h1>
      <p className="text-center text-gray-600">
      You need to pass the subject matter expert test to contribute questions
      </p><div className='flex  justify-center items-center'>
      <button className="bg-blue-500 hover:bg-blue-700 flex items-center text-white font-bold py-2 px-4 rounded mt-4" 
      onClick={()=>setStartExam(true)}>Start Exam</button>
      </div>
      </div>
    </div>:<>
       <div className="container mx-auto mt-4 p-4">
      <h1 className="text-2xl font-semibold mb-4 text-center">MCQ Subject Matter Expert Test</h1>
      <p className="text-center text-gray-600">
        Subject: Computer Science
      </p>
      <p className="text-center text-gray-600">
        Select the correct answer for each question
      </p>
      <MCQExam />
    </div></>}
    </div>
  )
}
