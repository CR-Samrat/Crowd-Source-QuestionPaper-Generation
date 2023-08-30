'use client'
import Head from 'next/head'
import React from 'react'
import {  Menu, Moon, X,  Database, Users, FileOutput, ArrowRight } from 'lucide-react'
import Link from 'next/link'
const menuItems = [
  {
    name: 'Home',
    href: '#',
  },
  {
    name: 'About',
    href: '#',
  },
  {
    name: 'Contact',
    href: '#',
  },
  {
    name: 'Blogs',
    href: '#',
  },
]

export default function LandinngPageTwo() {
  // const [isMenuOpen, setIsMenuOpen] = React.useState(false)
  const [hover, setHover] = React.useState(false)
  const clickhover = () => {
    setHover(!hover)
  }
  const [subject, setSubject] = React.useState('')

  return (
    <div className="w-full">
      <Head>
        <title>CrowdQuesTech</title>
         </Head>
      {/* Hero Section */}
      <div className="relative w-full bg-[url('/bg-image2.jpg')] bg-cover bg-center">
        <div className="mx-auto max-w-7xl lg:px-8">
          <div className="flex flex-col justify-center px-4 py-10 lg:px-6">
            
            <h1 className="mt-8 max-w-4xl text-3xl font-bold tracking-tight text-black md:text-4xl lg:text-6xl">
            Welcome to CrowdQuesTech
            </h1>
            {/* <p className="mt-8 max-w-3xl text-lg text-gray-700">
              Lorem ipsum dolor sit amet consectetur adipisicing elit. Officia ipsam nulla aperiam
              quo possimus, nihil molestiae modi tenetur esse qui repudiandae cum fuga aspernatur
              ea?
            </p> */}
            <div className="mt-8">
              <Link href="/contribute">
              <button
                type="button" 
                className="rounded-md flex bg-black px-3 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-black/80 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-black"
              >
                Contribute
                <ArrowRight className=' ms-2'/>
              </button>
              </Link>
            </div>
          </div>
          {/* <div className="rounded-lg bg-gray-200 p-4">
            <img
              className="aspect-[3/2] w-full rounded-lg bg-gray-50 object-cover lg:aspect-auto lg:h-[500px] lg:object-center"
              src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MjF8fHBlb3BsZXxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=800&q=60"
              alt=""
            />
          </div> */}
        </div>
      </div>
      <div className='container ms-5'>
      <h1 className='text-3xl font-semibold text-black mt-10 mb-10'>
          Create Question paper
      </h1>
      <p className='text-lg text-gray-700 mb-10'>
      Empower your education system with 
      a crowd-sourced question bank and elevate your assessment strategies to new heights.
      </p>
      <div className='flex flex-col items-center'>
      <button className='bg-black text-white px-4 py-2 rounded-md relative flex' onClick={clickhover}>
          <Link href="/generateq" className='flex items-center '>
              Generate Question Paper 
              <ArrowRight className=' ms-2'/>
          </Link>
  
      </button>
      </div>
      </div>
      {/* {hover && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white rounded">
          <button
              className=" px-4 py-2  text-black rounded float-right mt-0"
              onClick={clickhover}
            >
              X
            </button>
            <div className='mt-4 px-6 py-4'>
            <select className='border-2 border-black rounded-md px-2 py-1' onChange={(e)=>setSubject(e.target.value)}>
              <option value=''>Select Subject</option>
              <option value='Maths'>Maths</option>
              <option value='Physics'>Computer Science</option>
              <option value='Chemistry'>Chemistry</option>
            </select>
            <br/>
            <Link href="/generateq" className=' underline'>click to proceed</Link>
            </div>
          </div>
        </div>
      )} */}
      {/* Features Section */}
      <div className="mx-auto my-12 max-w-7xl px-4 sm:px-6 md:my-24 lg:my-32 lg:px-8">
        
        <div className="mt-12 grid grid-cols-1 gap-y-8 text-center sm:grid-cols-2 sm:gap-12 lg:grid-cols-3">
          <div>
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gray-100">
              <Users className="h-9 w-9 text-gray-700" />
            </div>
            <h3 className="mt-8 text-lg font-semibold text-black">Collect Questions from the Crowd</h3>
          </div>
          <div>
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gray-100">
              <Database className="h-9 w-9 text-gray-700" />
            </div>
            <h3 className="mt-8 text-lg font-semibold text-black">Build an Extensive Question Bank</h3>
            
          </div>
          <div>
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gray-100">
              <FileOutput className="h-9 w-9 text-gray-700" />
              
            </div>
            <h3 className="mt-8 text-lg font-semibold text-black">Create Customized Question Papers</h3>
          </div>
          
        </div>
      </div>
      {/* FAQs */}
      <section className="mx-auto max-w-7xl bg-gray-50 px-2 py-10 md:px-0">
        <div>
          <div className="mx-auto max-w-2xl lg:text-center">
            {/* <h2 className="text-3xl font-bold leading-tight text-black sm:text-4xl lg:text-5xl">
              Frequently Asked Questions
            </h2>
            <p className="mt-4 max-w-xl text-base leading-relaxed text-gray-600 lg:mx-auto">
              Lorem ipsum, dolor sit amet consectetur adipisicing elit. Facere, assumenda
            </p> */}
          </div>
          <div className="mx-auto mt-8 grid max-w-3xl grid-cols-1 gap-6 md:mt-16 md:grid-cols-2">
            {/* {Array.from({ length: 4 }).map((_, i) => (
              <div key={i}>
                <h2 className="text-xl font-semibold text-black">How do I get started?</h2>
                <p className="mt-6 text-sm leading-6 tracking-wide text-gray-500">
                  Lorem ipsum dolor sit amet consectetur adipisicing elit. Repellat aliquam adipisci
                  iusto aperiam? Sint asperiores sequi nobis inventore ratione deleniti?
                </p>
              </div>
            ))} */}
          </div>
          {/* <p className="mt-10 text-center text-gray-600">
            Can&apos;t find what you&apos;re looking for?{' '}
            <a href="#" title="" className="black font-semibold hover:underline">
              Contact us
            </a>
          </p> */}
        </div>
      </section>
      {/* Pricing Section */}
      
      
      {/* footer */}
      
    </div>
  )
}
